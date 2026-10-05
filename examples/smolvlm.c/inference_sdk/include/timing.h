#ifndef INFERENCE_SDK_TIMING_H
#define INFERENCE_SDK_TIMING_H

#if defined(TENSOR_TIMING_ENABLE)

#include <stdio.h>

#if defined(__APPLE__)
#include <mach/mach_time.h>
static inline double timing_now_ms(void) {
    static mach_timebase_info_data_t timebase;
    if (timebase.denom == 0) mach_timebase_info(&timebase);
    return (double)mach_absolute_time() * (double)timebase.numer /
           (double)timebase.denom / 1000000.0;
}
#else
#define _POSIX_C_SOURCE 200809L
#include <time.h>
static inline double timing_now_ms(void) {
    struct timespec value;
    if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) return 0.0;
    return (double)value.tv_sec * 1000.0 + (double)value.tv_nsec / 1000000.0;
}
#endif

static inline int timing_enabled(void) {
    return 1;
}

static inline double timing_start(void) {
    return timing_enabled() ? timing_now_ms() : 0.0;
}

static inline void timing_report(const char *phase, double started_ms) {
    if (started_ms != 0.0)
        fprintf(stderr, "timing_%s_ms=%.3f\n", phase, timing_now_ms() - started_ms);
}

static inline void timing_report_elapsed(const char *phase, double elapsed_ms) {
    if (timing_enabled())
        fprintf(stderr, "timing_%s_ms=%.3f\n", phase, elapsed_ms);
}

#else

#define timing_enabled() 0
#define timing_now_ms() 0.0
#define timing_start() 0.0
#define timing_report(phase, started_ms) ((void)sizeof(phase), (void)(started_ms))
#define timing_report_elapsed(phase, elapsed_ms) ((void)sizeof(phase), (void)(elapsed_ms))

#endif

#endif
