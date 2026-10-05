#include "benchmark.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    size_t prepare_calls;
    size_t run_calls;
    size_t sync_calls;
    size_t cleanup_calls;
    double fake_clock_ms;
    int fail;
} WorkloadState;

static int prepare(void *context, char *error, size_t capacity) {
    (void)error; (void)capacity;
    ((WorkloadState *)context)->prepare_calls++;
    return 1;
}

static int run(void *context, double *units, char *error, size_t capacity) {
    WorkloadState *state = context;
    state->run_calls++;
    if (state->fail) {
        snprintf(error, capacity, "requested failure");
        return 0;
    }
    *units = 3.0;
    return 1;
}

static int synchronize(void *context, char *error, size_t capacity) {
    (void)error; (void)capacity;
    ((WorkloadState *)context)->sync_calls++;
    return 1;
}

static double fake_clock(void *context) {
    WorkloadState *state = context;
    const double result = state->fake_clock_ms;
    state->fake_clock_ms += 1.0;
    return result;
}

static void cleanup(void *context) {
    ((WorkloadState *)context)->cleanup_calls++;
}

int main(void) {
    WorkloadState state = {0};
    const BenchmarkWorkload workload = {
        .name = "synthetic\"workload", .unit = "items", .context = &state,
        .prepare = prepare, .run = run, .cleanup = cleanup
    };
    const BenchmarkConfig config = {
        .warmup_iterations = 2, .measured_iterations = 5,
        .platform = "test", .backend = "fake", .device = "host",
        .platform_context = &state, .synchronize = synchronize,
        .clock_ms = fake_clock
    };
    BenchmarkSession *session = NULL;
    char error[128] = {0};
    assert(benchmark_session_create(&config, &workload, &session,
                                    error, sizeof(error)));
    assert(benchmark_session_report(session) == NULL);
    if (!benchmark_session_run(session, error, sizeof(error))) {
        fprintf(stderr, "benchmark run failed: %s\n", error);
        abort();
    }
    const BenchmarkReport *report = benchmark_session_report(session);
    assert(report != NULL);
    assert(report->warmup_iterations == 2 && report->measured_iterations == 5);
    assert(report->min_latency_ms > 0.0);
    assert(report->max_latency_ms >= report->median_latency_ms);
    assert(report->p95_latency_ms >= report->min_latency_ms);
    assert(report->total_work_units == 15.0);
    assert(report->throughput_per_second > 0.0);
    assert(state.prepare_calls == 1 && state.run_calls == 7);
    assert(state.sync_calls == 10 && state.cleanup_calls == 1);
    assert(!benchmark_session_run(session, error, sizeof(error)));
    char *json = NULL;
    assert(benchmark_report_to_json(report, &json, error, sizeof(error)));
    assert(strstr(json, "synthetic\\\"workload") != NULL);
    assert(strstr(json, "\"measured_iterations\":5") != NULL);
    free(json);
    benchmark_session_destroy(session);

    state = (WorkloadState){.fail = 1};
    assert(benchmark_session_create(&config, &workload, &session,
                                    error, sizeof(error)));
    assert(!benchmark_session_run(session, error, sizeof(error)));
    assert(strcmp(error, "requested failure") == 0);
    assert(state.cleanup_calls == 1);
    benchmark_session_destroy(session);

    puts("benchmark SDK tests passed");
    return 0;
}
