#include "benchmark.h"

#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static int compare_double(const void *left, const void *right) {
    const double a = *(const double *)left;
    const double b = *(const double *)right;
    return (a > b) - (a < b);
}

int main(int argc, char **argv) {
    if (argc < 6) {
        fprintf(stderr, "usage: %s NAME BACKEND WARMUPS WORK_UNITS LATENCY_MS...\n", argv[0]);
        return 2;
    }
    char *end = NULL;
    errno = 0;
    const unsigned long warmups = strtoul(argv[3], &end, 10);
    if (errno || end == argv[3] || *end != '\0') return 2;
    errno = 0;
    const double work_units = strtod(argv[4], &end);
    if (errno || end == argv[4] || *end != '\0' || !isfinite(work_units) || work_units < 0.0) return 2;

    const size_t count = (size_t)(argc - 5);
    double *latencies = calloc(count, sizeof(*latencies));
    double *sorted = calloc(count, sizeof(*sorted));
    double *units = calloc(count, sizeof(*units));
    if (latencies == NULL || sorted == NULL || units == NULL) {
        free(latencies); free(sorted); free(units);
        return 1;
    }
    double total_ms = 0.0;
    for (size_t i = 0; i < count; ++i) {
        errno = 0;
        latencies[i] = strtod(argv[i + 5], &end);
        if (errno || end == argv[i + 5] || *end != '\0' || !isfinite(latencies[i]) || latencies[i] < 0.0) {
            fprintf(stderr, "invalid latency: %s\n", argv[i + 5]);
            free(latencies); free(sorted); free(units);
            return 2;
        }
        sorted[i] = latencies[i];
        units[i] = work_units;
        total_ms += latencies[i];
    }
    qsort(sorted, count, sizeof(*sorted), compare_double);
    const BenchmarkReport report = {
        .workload_name = argv[1], .unit = "decision", .platform = "",
        .backend = argv[2], .device = "", .warmup_iterations = (size_t)warmups,
        .measured_iterations = count, .latency_ms = latencies, .work_units = units,
        .min_latency_ms = sorted[0], .mean_latency_ms = total_ms / (double)count,
        .median_latency_ms = count % 2 ? sorted[count / 2] :
            (sorted[count / 2 - 1] + sorted[count / 2]) * 0.5,
        .p95_latency_ms = sorted[((count - 1) * 95) / 100],
        .max_latency_ms = sorted[count - 1], .total_work_units = work_units * (double)count,
        .throughput_per_second = total_ms > 0.0 ? work_units * (double)count * 1000.0 / total_ms : 0.0
    };
    char *text = NULL;
    char error[256] = {0};
    const int ok = benchmark_report_to_text(&report, &text, error, sizeof(error));
    if (!ok) fprintf(stderr, "benchmark_report_to_text: %s\n", error);
    else { puts(text); free(text); }
    free(latencies); free(sorted); free(units);
    return ok ? 0 : 1;
}
