#ifndef BENCHMARK_H
#define BENCHMARK_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct BenchmarkSession BenchmarkSession;

typedef int (*BenchmarkPrepareFn)(void *context, char *error, size_t error_capacity);
typedef int (*BenchmarkRunFn)(void *context, double *work_units,
                              char *error, size_t error_capacity);
typedef int (*BenchmarkSynchronizeFn)(void *context, char *error,
                                      size_t error_capacity);
typedef double (*BenchmarkClockFn)(void *context);
typedef void (*BenchmarkCleanupFn)(void *context);

typedef struct {
    const char *name;
    const char *unit;
    void *context;
    BenchmarkPrepareFn prepare;
    BenchmarkRunFn run;
    BenchmarkCleanupFn cleanup;
} BenchmarkWorkload;

/* Synchronization runs immediately before and after every timed iteration. */
typedef struct {
    size_t warmup_iterations;
    size_t measured_iterations;
    const char *platform;
    const char *backend;
    const char *device;
    void *platform_context;
    BenchmarkSynchronizeFn synchronize;
    BenchmarkClockFn clock_ms;
} BenchmarkConfig;

typedef struct {
    const char *workload_name;
    const char *unit;
    const char *platform;
    const char *backend;
    const char *device;
    size_t warmup_iterations;
    size_t measured_iterations;
    const double *latency_ms;
    const double *work_units;
    double min_latency_ms;
    double mean_latency_ms;
    double median_latency_ms;
    double p95_latency_ms;
    double max_latency_ms;
    double total_work_units;
    double throughput_per_second;
} BenchmarkReport;

int benchmark_session_create(const BenchmarkConfig *config,
                             const BenchmarkWorkload *workload,
                             BenchmarkSession **result,
                             char *error, size_t error_capacity);
int benchmark_session_run(BenchmarkSession *session,
                          char *error, size_t error_capacity);
const BenchmarkReport *benchmark_session_report(const BenchmarkSession *session);
void benchmark_session_destroy(BenchmarkSession *session);

/* Allocates a compact JSON representation; release it with free(). */
int benchmark_report_to_json(const BenchmarkReport *report, char **json,
                             char *error, size_t error_capacity);

/* Allocates a human-readable summary; release it with free(). */
int benchmark_report_to_text(const BenchmarkReport *report, char **text,
                             char *error, size_t error_capacity);

#ifdef __cplusplus
}
#endif

#endif
