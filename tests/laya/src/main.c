#include "laya_model.h"

#include "benchmark.h"
#include "tensor_device.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct {
    const char *model_directory;
    const char *backend;
    LayaModel *model;
    LayaPrediction prediction;
} Workload;

static int prepare(void *context, char *error, size_t capacity) {
    Workload *workload = context;
    if (!laya_model_open(workload->model_directory, &workload->model, error, capacity)) return 0;
    workload->backend = laya_model_backend(workload->model);
    return 1;
}

static int run(void *context, double *work_units, char *error, size_t capacity) {
    Workload *workload = context;
    if (!laya_model_predict(workload->model, &workload->prediction, error, capacity)) return 0;
    *work_units = 3.0;
    return 1;
}

static int synchronize(void *context, char *error, size_t capacity) {
    (void)context;
    return tensor_device_synchronize(error, capacity);
}

static void cleanup(void *context) {
    Workload *workload = context;
    laya_model_close(workload->model);
    workload->model = NULL;
}

static int parse_count(const char *text, size_t *value) {
    char *end = NULL;
    errno = 0;
    const unsigned long long parsed = strtoull(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || parsed > SIZE_MAX) return 0;
    *value = (size_t)parsed;
    return 1;
}

static size_t winner(const LayaDecision *decision) {
    size_t best = 0;
    for (size_t i = 1; i < decision->option_count; ++i)
        if (decision->probabilities[i] > decision->probabilities[best]) best = i;
    return best;
}

static void print_decisions_text(const LayaPrediction *prediction) {
    puts("decisions:");
    for (size_t i = 0; i < 3; ++i) {
        const LayaDecision *d = &prediction->decisions[i];
        printf("  %s (%s, %zu input tokens): %s\n", d->name, d->type,
               d->input_tokens, d->labels[winner(d)]);
        for (size_t j = 0; j < d->option_count; ++j)
            printf("    %s: %.6f\n", d->labels[j], d->probabilities[j]);
    }
}

static void print_decisions_json(const LayaPrediction *prediction) {
    printf("\"decisions\":[");
    for (size_t i = 0; i < 3; ++i) {
        const LayaDecision *d = &prediction->decisions[i];
        if (i != 0) putchar(',');
        printf("{\"name\":\"%s\",\"type\":\"%s\",\"input_tokens\":%zu,\"answer\":\"%s\",\"probabilities\":{",
               d->name, d->type, d->input_tokens, d->labels[winner(d)]);
        for (size_t j = 0; j < d->option_count; ++j) {
            if (j != 0) putchar(',');
            printf("\"%s\":%.9g", d->labels[j], d->probabilities[j]);
        }
        printf("}}");
    }
    printf("]");
}

static void usage(const char *program) {
    fprintf(stderr,
        "Usage: %s <multilingual-model-dir> [runs=5] [warmup-runs=2] [text|json]\n"
        "GPU-only Laya multilingual typed-decision example (Metal/CUDA FP16).\n",
        program);
}

int main(int argc, char **argv) {
    if (argc < 2 || argc > 5 || strcmp(argv[1], "--help") == 0 || strcmp(argv[1], "-h") == 0) {
        usage(argv[0]);
        return argc >= 2 && (strcmp(argv[1], "--help") == 0 || strcmp(argv[1], "-h") == 0) ? 0 : 2;
    }
    const char *backend_env = getenv("TENSOR_BACKEND");
    if (backend_env == NULL || (strcmp(backend_env, "metal") != 0 && strcmp(backend_env, "cuda") != 0)) {
        fprintf(stderr, "TENSOR_BACKEND must explicitly be metal or cuda; CPU inference is disabled.\n");
        return 2;
    }
    size_t runs = 5, warmups = 2;
    if ((argc >= 3 && !parse_count(argv[2], &runs)) ||
        (argc >= 4 && !parse_count(argv[3], &warmups)) || runs == 0) {
        fprintf(stderr, "runs must be positive and warmup-runs must be non-negative integers\n");
        return 2;
    }
    const char *format = argc >= 5 ? argv[4] : "text";
    if (strcmp(format, "text") != 0 && strcmp(format, "json") != 0) {
        fprintf(stderr, "format must be text or json\n");
        return 2;
    }
    Workload workload = {.model_directory = argv[1]};
    const BenchmarkWorkload definition = {
        .name = "laya_typed_decisions",
        .unit = "decision",
        .context = &workload,
        .prepare = prepare,
        .run = run,
        .cleanup = cleanup
    };
    const BenchmarkConfig config = {
        .warmup_iterations = warmups,
        .measured_iterations = runs,
        .platform = "gpu",
        .backend = backend_env,
        .device = "",
        .synchronize = synchronize
    };
    char error[512] = {0};
    BenchmarkSession *session = NULL;
    if (!benchmark_session_create(&config, &definition, &session, error, sizeof(error)) ||
        !benchmark_session_run(session, error, sizeof(error))) {
        fprintf(stderr, "Laya inference failed: %s\n", error[0] ? error : "unknown error");
        benchmark_session_destroy(session);
        cleanup(&workload);
        return 1;
    }
    const BenchmarkReport *report = benchmark_session_report(session);
    if (strcmp(format, "json") == 0) {
        char *json = NULL;
        if (!benchmark_report_to_json(report, &json, error, sizeof(error))) {
            fprintf(stderr, "benchmark JSON formatting failed: %s\n", error);
            benchmark_session_destroy(session); cleanup(&workload); return 1;
        }
        printf("{\"benchmark\":%s,", json);
        print_decisions_json(&workload.prediction);
        printf("}\n");
        free(json);
    } else {
        char *text = NULL;
        if (!benchmark_report_to_text(report, &text, error, sizeof(error))) {
            fprintf(stderr, "benchmark text formatting failed: %s\n", error);
            benchmark_session_destroy(session); cleanup(&workload); return 1;
        }
        puts(text);
        printf("model: %s\n", argv[1]);
        printf("engine: dsinfra native FP16 (%s)\n", workload.backend);
        free(text);
        print_decisions_text(&workload.prediction);
    }
    benchmark_session_destroy(session);
    cleanup(&workload);
    return 0;
}
