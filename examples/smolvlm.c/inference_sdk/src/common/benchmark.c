#define _POSIX_C_SOURCE 200809L

#include "benchmark.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__APPLE__)
#include <mach/mach_time.h>
#else
#include <time.h>
#endif

#define BENCHMARK_MAX_ITERATIONS 100000u
#define BENCHMARK_ERROR_CAPACITY 512u

struct BenchmarkSession {
    BenchmarkConfig config;
    BenchmarkWorkload workload;
    char *workload_name;
    char *unit;
    char *platform;
    char *backend;
    char *device;
    double *latency_ms;
    double *work_units;
    BenchmarkReport report;
    int has_run;
};

static void set_error(char *error, size_t capacity, const char *message) {
    if (error != NULL && capacity > 0) snprintf(error, capacity, "%s", message);
}

static char *copy_string(const char *value, const char *fallback) {
    if (value == NULL) value = fallback;
    const size_t length = strlen(value);
    char *copy = malloc(length + 1);
    if (copy != NULL) memcpy(copy, value, length + 1);
    return copy;
}

static double monotonic_ms(void) {
#if defined(__APPLE__)
    static mach_timebase_info_data_t timebase;
    if (timebase.denom == 0) mach_timebase_info(&timebase);
    return (double)mach_absolute_time() * (double)timebase.numer /
           (double)timebase.denom / 1000000.0;
#else
    struct timespec value;
    if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) return 0.0;
    return (double)value.tv_sec * 1000.0 + (double)value.tv_nsec / 1000000.0;
#endif
}

static int synchronize(BenchmarkSession *session, char *error, size_t capacity) {
    if (session->config.synchronize == NULL) return 1;
    return session->config.synchronize(session->config.platform_context,
                                       error, capacity);
}

static double session_clock_ms(BenchmarkSession *session) {
    return session->config.clock_ms == NULL ? monotonic_ms() :
        session->config.clock_ms(session->config.platform_context);
}

static int execute_once(BenchmarkSession *session, double *units,
                        char *error, size_t capacity) {
    double value = 0.0;
    if (!session->workload.run(session->workload.context, &value, error, capacity))
        return 0;
    if (!isfinite(value) || value < 0.0) {
        set_error(error, capacity, "workload returned invalid work units");
        return 0;
    }
    *units = value;
    return 1;
}

int benchmark_session_create(const BenchmarkConfig *config,
                             const BenchmarkWorkload *workload,
                             BenchmarkSession **result,
                             char *error, size_t error_capacity) {
    if (result != NULL) *result = NULL;
    if (config == NULL || workload == NULL || result == NULL ||
        workload->name == NULL || workload->name[0] == '\0' ||
        workload->run == NULL || config->measured_iterations == 0 ||
        config->measured_iterations > BENCHMARK_MAX_ITERATIONS ||
        config->warmup_iterations > BENCHMARK_MAX_ITERATIONS) {
        set_error(error, error_capacity, "invalid benchmark configuration or workload");
        return 0;
    }
    BenchmarkSession *session = calloc(1, sizeof(*session));
    if (session == NULL) {
        set_error(error, error_capacity, "out of memory creating benchmark session");
        return 0;
    }
    session->config = *config;
    session->workload = *workload;
    session->workload_name = copy_string(workload->name, "");
    session->unit = copy_string(workload->unit, "iteration");
    session->platform = copy_string(config->platform, "unspecified");
    session->backend = copy_string(config->backend, "unspecified");
    session->device = copy_string(config->device, "unspecified");
    session->latency_ms = calloc(config->measured_iterations, sizeof(double));
    session->work_units = calloc(config->measured_iterations, sizeof(double));
    if (session->workload_name == NULL || session->unit == NULL ||
        session->platform == NULL || session->backend == NULL ||
        session->device == NULL || session->latency_ms == NULL ||
        session->work_units == NULL) {
        benchmark_session_destroy(session);
        set_error(error, error_capacity, "out of memory preparing benchmark session");
        return 0;
    }
    session->config.platform = session->platform;
    session->config.backend = session->backend;
    session->config.device = session->device;
    session->workload.name = session->workload_name;
    session->workload.unit = session->unit;
    *result = session;
    return 1;
}

static int compare_double(const void *left, const void *right) {
    const double a = *(const double *)left, b = *(const double *)right;
    return a < b ? -1 : a > b ? 1 : 0;
}

int benchmark_session_run(BenchmarkSession *session,
                          char *error, size_t error_capacity) {
    if (session == NULL || session->has_run) {
        set_error(error, error_capacity, "benchmark session is null or has already run");
        return 0;
    }
    session->has_run = 1;
    int prepared = 0, ok = 1;
    char callback_error[BENCHMARK_ERROR_CAPACITY] = {0};
    if (session->workload.prepare != NULL) {
        ok = session->workload.prepare(session->workload.context,
                                       callback_error, sizeof(callback_error));
        prepared = ok;
        if (!ok && callback_error[0] == '\0')
            snprintf(callback_error, sizeof(callback_error), "workload preparation failed");
    } else prepared = 1;

    for (size_t i = 0; ok && i < session->config.warmup_iterations; ++i) {
        ok = execute_once(session, &session->work_units[0], callback_error,
                          sizeof(callback_error));
    }
    double total_ms = 0.0, total_units = 0.0;
    for (size_t i = 0; ok && i < session->config.measured_iterations; ++i) {
        if (!synchronize(session, callback_error, sizeof(callback_error))) {
            ok = 0;
            break;
        }
        const double started = session_clock_ms(session);
        ok = isfinite(started) && execute_once(session, &session->work_units[i],
                                                callback_error, sizeof(callback_error));
        if (ok) ok = synchronize(session, callback_error, sizeof(callback_error));
        const double finished = session_clock_ms(session);
        const double elapsed = finished - started;
        if (ok && (!isfinite(elapsed) || elapsed <= 0.0)) {
            set_error(callback_error, sizeof(callback_error), "invalid benchmark clock interval");
            ok = 0;
        }
        if (ok) {
            session->latency_ms[i] = elapsed;
            total_ms += elapsed;
            total_units += session->work_units[i];
        }
    }
    if (prepared && session->workload.cleanup != NULL)
        session->workload.cleanup(session->workload.context);
    if (!ok) {
        set_error(error, error_capacity, callback_error[0] == '\0' ?
                  "benchmark execution failed" : callback_error);
        return 0;
    }

    const size_t count = session->config.measured_iterations;
    double *sorted = malloc(count * sizeof(double));
    if (sorted == NULL) {
        set_error(error, error_capacity, "out of memory calculating benchmark statistics");
        return 0;
    }
    memcpy(sorted, session->latency_ms, count * sizeof(double));
    qsort(sorted, count, sizeof(double), compare_double);
    session->report = (BenchmarkReport){
        .workload_name = session->workload_name,
        .unit = session->unit,
        .platform = session->platform,
        .backend = session->backend,
        .device = session->device,
        .warmup_iterations = session->config.warmup_iterations,
        .measured_iterations = count,
        .latency_ms = session->latency_ms,
        .work_units = session->work_units,
        .min_latency_ms = sorted[0],
        .mean_latency_ms = total_ms / (double)count,
        .median_latency_ms = count % 2 ? sorted[count / 2] :
            (sorted[count / 2 - 1] + sorted[count / 2]) * 0.5,
        .p95_latency_ms = sorted[((count - 1) * 95) / 100],
        .max_latency_ms = sorted[count - 1],
        .total_work_units = total_units,
        .throughput_per_second = total_ms > 0.0 ? total_units * 1000.0 / total_ms : 0.0
    };
    free(sorted);
    return 1;
}

const BenchmarkReport *benchmark_session_report(const BenchmarkSession *session) {
    return session != NULL && session->report.latency_ms != NULL ? &session->report : NULL;
}

void benchmark_session_destroy(BenchmarkSession *session) {
    if (session == NULL) return;
    free(session->workload_name);
    free(session->unit);
    free(session->platform);
    free(session->backend);
    free(session->device);
    free(session->latency_ms);
    free(session->work_units);
    free(session);
}

typedef struct { char *data; size_t length, capacity; } JsonBuilder;

static int json_reserve(JsonBuilder *builder, size_t extra) {
    if (extra > SIZE_MAX - builder->length - 1) return 0;
    const size_t needed = builder->length + extra + 1;
    if (needed <= builder->capacity) return 1;
    size_t capacity = builder->capacity == 0 ? 256 : builder->capacity;
    while (capacity < needed) {
        if (capacity > SIZE_MAX / 2) { capacity = needed; break; }
        capacity *= 2;
    }
    char *grown = realloc(builder->data, capacity);
    if (grown == NULL) return 0;
    builder->data = grown;
    builder->capacity = capacity;
    return 1;
}

static int json_append(JsonBuilder *builder, const char *text) {
    const size_t length = strlen(text);
    if (!json_reserve(builder, length)) return 0;
    memcpy(builder->data + builder->length, text, length + 1);
    builder->length += length;
    return 1;
}

static int json_append_string(JsonBuilder *builder, const char *text) {
    if (!json_append(builder, "\"")) return 0;
    for (const unsigned char *p = (const unsigned char *)text; *p != 0; ++p) {
        char escaped[7];
        if (*p == '"' || *p == '\\') {
            escaped[0] = '\\'; escaped[1] = (char)*p; escaped[2] = '\0';
        } else if (*p < 0x20) snprintf(escaped, sizeof(escaped), "\\u%04x", *p);
        else { escaped[0] = (char)*p; escaped[1] = '\0'; }
        if (!json_append(builder, escaped)) return 0;
    }
    return json_append(builder, "\"");
}

static int json_append_number(JsonBuilder *builder, const char *format,
                              double value) {
    char number[64];
    const int length = snprintf(number, sizeof(number), format, value);
    return length > 0 && (size_t)length < sizeof(number) &&
        json_append(builder, number);
}

int benchmark_report_to_json(const BenchmarkReport *report, char **json,
                             char *error, size_t error_capacity) {
    if (json != NULL) *json = NULL;
    if (report == NULL || json == NULL || report->latency_ms == NULL ||
        report->work_units == NULL || report->measured_iterations == 0) {
        set_error(error, error_capacity, "invalid benchmark report");
        return 0;
    }
    JsonBuilder builder = {0};
    int ok = json_append(&builder, "{\"workload\":") &&
        json_append_string(&builder, report->workload_name) &&
        json_append(&builder, ",\"unit\":") && json_append_string(&builder, report->unit) &&
        json_append(&builder, ",\"platform\":") && json_append_string(&builder, report->platform) &&
        json_append(&builder, ",\"backend\":") && json_append_string(&builder, report->backend) &&
        json_append(&builder, ",\"device\":") && json_append_string(&builder, report->device);
    char integer[64];
    snprintf(integer, sizeof(integer), ",\"warmup_iterations\":%zu,\"measured_iterations\":%zu,\"latency_ms\":[",
             report->warmup_iterations, report->measured_iterations);
    ok = ok && json_append(&builder, integer);
    for (size_t i = 0; ok && i < report->measured_iterations; ++i) {
        if (i != 0) ok = json_append(&builder, ",");
        ok = ok && json_append_number(&builder, "%.9g", report->latency_ms[i]);
    }
    ok = ok && json_append(&builder, "],\"work_units\":[");
    for (size_t i = 0; ok && i < report->measured_iterations; ++i) {
        if (i != 0) ok = json_append(&builder, ",");
        ok = ok && json_append_number(&builder, "%.9g", report->work_units[i]);
    }
    ok = ok && json_append(&builder, "],\"min_latency_ms\":") &&
        json_append_number(&builder, "%.9g", report->min_latency_ms) &&
        json_append(&builder, ",\"mean_latency_ms\":") &&
        json_append_number(&builder, "%.9g", report->mean_latency_ms) &&
        json_append(&builder, ",\"median_latency_ms\":") &&
        json_append_number(&builder, "%.9g", report->median_latency_ms) &&
        json_append(&builder, ",\"p95_latency_ms\":") &&
        json_append_number(&builder, "%.9g", report->p95_latency_ms) &&
        json_append(&builder, ",\"max_latency_ms\":") &&
        json_append_number(&builder, "%.9g", report->max_latency_ms) &&
        json_append(&builder, ",\"total_work_units\":") &&
        json_append_number(&builder, "%.9g", report->total_work_units) &&
        json_append(&builder, ",\"throughput_per_second\":") &&
        json_append_number(&builder, "%.9g", report->throughput_per_second) &&
        json_append(&builder, "}");
    if (!ok) {
        free(builder.data);
        set_error(error, error_capacity, "out of memory serializing benchmark report");
        return 0;
    }
    *json = builder.data;
    return 1;
}
