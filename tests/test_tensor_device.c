#define _POSIX_C_SOURCE 200809L

#include "tensor_buffer.h"
#include "tensor_device.h"
#include "tensor_ops.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define TEST_BYTES (16u * 1024u * 1024u)
#define TEST_ROUNDS 4u
#define TEST_WORKING_SET_BYTES (TEST_BYTES * TEST_ROUNDS)
#define PERFORMANCE_SAMPLES 5u

static double now_seconds(void) {
    struct timespec value;
    if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) return 0.0;
    return (double)value.tv_sec + (double)value.tv_nsec / 1.0e9;
}

static double median_seconds(const double *samples, size_t count) {
    double sorted[PERFORMANCE_SAMPLES];
    if (samples == NULL || count == 0 || count > PERFORMANCE_SAMPLES) return 0.0;
    memcpy(sorted, samples, count * sizeof(*sorted));
    for (size_t i = 1; i < count; ++i) {
        const double value = sorted[i];
        size_t j = i;
        while (j > 0 && sorted[j - 1] > value) {
            sorted[j] = sorted[j - 1];
            --j;
        }
        sorted[j] = value;
    }
    return sorted[count / 2];
}

static void sample_range(const double *samples, size_t count,
                         double *minimum, double *maximum) {
    *minimum = *maximum = count == 0 ? 0.0 : samples[0];
    for (size_t i = 1; i < count; ++i) {
        if (samples[i] < *minimum) *minimum = samples[i];
        if (samples[i] > *maximum) *maximum = samples[i];
    }
}

static int fail(const char *message, char *error) {
    fprintf(stderr, "test_tensor_device: %s%s%s\n", message,
            error != NULL && error[0] != '\0' ? ": " : "",
            error != NULL ? error : "");
    return 1;
}

int main(void) {
    if (!tensor_device_available()) {
        puts("SKIP: no configured GPU device is available");
        return 77;
    }

    char error[256] = {0};
    const uint64_t shape[] = {TEST_WORKING_SET_BYTES};
    TensorBuffer *source = NULL;
    TensorBuffer *destination = NULL;
    if (!tensor_buffer_create_device(TENSOR_DTYPE_U8, shape, 1, &source,
                                     error, sizeof(error)) ||
        !tensor_buffer_create_device(TENSOR_DTYPE_U8, shape, 1, &destination,
                                     error, sizeof(error))) {
        tensor_buffer_free(source);
        tensor_buffer_free(destination);
        return fail("device allocation failed", error);
    }
    if (tensor_buffer_storage(source) != TENSOR_BUFFER_STORAGE_DEVICE ||
        tensor_buffer_data(source) != NULL ||
        tensor_buffer_byte_length(source) != TEST_WORKING_SET_BYTES) {
        tensor_buffer_free(source);
        tensor_buffer_free(destination);
        return fail("device tensor storage contract failed", NULL);
    }

    uint8_t *host_source = malloc(TEST_WORKING_SET_BYTES);
    uint8_t *host_result = malloc(TEST_BYTES);
    if (host_source == NULL || host_result == NULL) {
        free(host_source);
        free(host_result);
        tensor_buffer_free(source);
        tensor_buffer_free(destination);
        return fail("host staging allocation failed", NULL);
    }
    for (size_t i = 0; i < TEST_WORKING_SET_BYTES; ++i)
        host_source[i] = (uint8_t)((i * 131u + (i >> 8) + 17u) & 0xffu);

    if (!tensor_buffer_upload(source, 0, host_source, TEST_BYTES,
                              error, sizeof(error)) ||
        !tensor_buffer_copy_device(destination, 0, source, 0, TEST_BYTES,
                                  error, sizeof(error)) ||
        !tensor_device_synchronize(error, sizeof(error)) ||
        !tensor_buffer_download(destination, 0, host_result, TEST_BYTES,
                                error, sizeof(error))) {
        free(host_source);
        free(host_result);
        tensor_buffer_free(source);
        tensor_buffer_free(destination);
        return fail("upload/copy/download failed", error);
    }
    if (memcmp(host_source, host_result, TEST_BYTES) != 0) {
        free(host_source);
        free(host_result);
        tensor_buffer_free(source);
        tensor_buffer_free(destination);
        return fail("GPU buffer round trip changed data", NULL);
    }

    /* Verify device-resident GEMM against a small, independently computed result. */
    {
        const uint64_t xs[] = {2, 3}, ws[] = {2, 3}, bs[] = {2}, ys[] = {2, 2};
        const float hx[] = {1, 2, 3, -1, 0.5f, 2};
        const float hw[] = {2, -1, 0.5f, -2, 3, 1};
        const float hb[] = {0.25f, -0.75f};
        const float expected[] = {1.75f, 6.25f, -1.25f, 4.75f};
        TensorBuffer *x = NULL, *w = NULL, *b = NULL, *y = NULL;
        float actual[4] = {0};
        int ok = tensor_buffer_create_device(TENSOR_DTYPE_F32, xs, 2, &x, error, sizeof(error)) &&
            tensor_buffer_create_device(TENSOR_DTYPE_F32, ws, 2, &w, error, sizeof(error)) &&
            tensor_buffer_create_device(TENSOR_DTYPE_F32, bs, 1, &b, error, sizeof(error)) &&
            tensor_buffer_create_device(TENSOR_DTYPE_F32, ys, 2, &y, error, sizeof(error)) &&
            tensor_buffer_upload(x, 0, hx, sizeof(hx), error, sizeof(error)) &&
            tensor_buffer_upload(w, 0, hw, sizeof(hw), error, sizeof(error)) &&
            tensor_buffer_upload(b, 0, hb, sizeof(hb), error, sizeof(error)) &&
            tensor_linear_f32(x, w, b, y, error, sizeof(error)) &&
            tensor_buffer_download(y, 0, actual, sizeof(actual), error, sizeof(error));
        for (size_t i = 0; ok && i < 4; ++i)
            if (actual[i] - expected[i] > 1e-5f || expected[i] - actual[i] > 1e-5f) ok = 0;
        tensor_buffer_free(x); tensor_buffer_free(w); tensor_buffer_free(b); tensor_buffer_free(y);
        if (!ok) {
            free(host_source); free(host_result);
            tensor_buffer_free(source); tensor_buffer_free(destination);
            return fail("device F32 linear disagrees with reference", error);
        }
        puts("GPU consistency: F32 linear+bias matched the host reference");
    }

    /* Verify asynchronous device F16 residual add against exact half values. */
    {
        const uint64_t shape[] = {4};
        const uint16_t left_values[] = {0x3c00, 0xc000, 0x4200, 0xc400};
        const uint16_t right_values[] = {0x4000, 0x3800, 0xbc00, 0x4400};
        const uint16_t expected[] = {0x4200, 0xbe00, 0x4000, 0x0000};
        uint16_t actual[4] = {0};
        TensorBuffer *left = NULL, *right = NULL, *output = NULL;
        int ok = tensor_buffer_create_device(TENSOR_DTYPE_F16, shape, 1, &left,
                    error, sizeof(error)) &&
            tensor_buffer_create_device(TENSOR_DTYPE_F16, shape, 1, &right,
                    error, sizeof(error)) &&
            tensor_buffer_create_device(TENSOR_DTYPE_F16, shape, 1, &output,
                    error, sizeof(error)) &&
            tensor_buffer_upload(left, 0, left_values, sizeof(left_values),
                    error, sizeof(error)) &&
            tensor_buffer_upload(right, 0, right_values, sizeof(right_values),
                    error, sizeof(error)) &&
            tensor_residual_add_f16(left, right, output, error, sizeof(error)) &&
            tensor_device_synchronize(error, sizeof(error)) &&
            tensor_buffer_download(output, 0, actual, sizeof(actual),
                    error, sizeof(error));
        const int matches = ok && memcmp(actual, expected, sizeof(expected)) == 0;
        tensor_buffer_free(left);
        tensor_buffer_free(right);
        tensor_buffer_free(output);
        if (!matches) {
            free(host_source); free(host_result);
            tensor_buffer_free(source); tensor_buffer_free(destination);
            return fail("device F16 residual add disagrees with reference", error);
        }
        puts("GPU consistency: asynchronous F16 residual add matched exact reference");
    }

    /* Queue a representative decode/prefill GEMM workload and synchronize once. */
    {
        enum { ROWS = 64, WIDTH = 256, ROUNDS = 32 };
        const uint64_t xs[] = {ROWS, WIDTH}, ws[] = {WIDTH, WIDTH}, ys[] = {ROWS, WIDTH};
        const size_t xn = (size_t)ROWS * WIDTH, wn = (size_t)WIDTH * WIDTH;
        float *hx = calloc(xn, sizeof(float)), *hw = calloc(wn, sizeof(float));
        float *hy = malloc(xn * sizeof(float));
        TensorBuffer *x = NULL, *w = NULL, *y = NULL;
        int ok = hx != NULL && hw != NULL && hy != NULL;
        for (size_t i = 0; ok && i < xn; ++i) hx[i] = (float)((int)(i % 17) - 8) * 0.001f;
        for (size_t i = 0; ok && i < wn; ++i) hw[i] = (float)((int)(i % 13) - 6) * 0.001f;
        ok = ok && tensor_buffer_create_device(TENSOR_DTYPE_F32, xs, 2, &x, error, sizeof(error)) &&
            tensor_buffer_create_device(TENSOR_DTYPE_F32, ws, 2, &w, error, sizeof(error)) &&
            tensor_buffer_create_device(TENSOR_DTYPE_F32, ys, 2, &y, error, sizeof(error)) &&
            tensor_buffer_upload(x, 0, hx, xn * sizeof(float), error, sizeof(error)) &&
            tensor_buffer_upload(w, 0, hw, wn * sizeof(float), error, sizeof(error));
        double elapsed_samples[PERFORMANCE_SAMPLES] = {0};
        if (ok) {
            /* Drop one warm-up sample, then report the median of five runs. */
            for (size_t sample = 0; sample <= PERFORMANCE_SAMPLES && ok; ++sample) {
                const double start = now_seconds();
                for (size_t round = 0; round < ROUNDS && ok; ++round)
                    ok = tensor_linear_f32(x, w, NULL, y, error, sizeof(error));
                ok = ok && tensor_buffer_download(y, 0, hy, xn * sizeof(float), error, sizeof(error));
                const double elapsed = now_seconds() - start;
                if (sample > 0) elapsed_samples[sample - 1] = elapsed;
            }
        }
        if (!ok) {
            free(hx); free(hw); free(hy);
            tensor_buffer_free(x); tensor_buffer_free(w); tensor_buffer_free(y);
            free(host_source); free(host_result);
            tensor_buffer_free(source); tensor_buffer_free(destination);
            return fail("GPU linear performance workload failed", error);
        }
        const double operations = 2.0 * ROWS * WIDTH * WIDTH * ROUNDS;
        double minimum = 0.0, maximum = 0.0;
        sample_range(elapsed_samples, PERFORMANCE_SAMPLES, &minimum, &maximum);
        const double median = median_seconds(elapsed_samples, PERFORMANCE_SAMPLES);
        printf("GPU F32 linear baseline: %.2f GFLOP/s median (%.3f-%.3f ms, "
               "%u samples; rows=%d, in=out=%d, %d queued rounds)\n",
               median > 0.0 ? operations / median / 1.0e9 : 0.0,
               minimum * 1000.0, maximum * 1000.0, PERFORMANCE_SAMPLES,
               ROWS, WIDTH, ROUNDS);
        free(hx); free(hw); free(hy);
        tensor_buffer_free(x); tensor_buffer_free(w); tensor_buffer_free(y);
    }

    double upload_samples[PERFORMANCE_SAMPLES] = {0};
    double copy_samples[PERFORMANCE_SAMPLES] = {0};
    for (size_t sample = 0; sample <= PERFORMANCE_SAMPLES; ++sample) {
        double start = now_seconds();
        for (size_t i = 0; i < TEST_ROUNDS; ++i) {
            const size_t offset = i * TEST_BYTES;
            if (!tensor_buffer_upload(source, offset, host_source + offset, TEST_BYTES,
                                      error, sizeof(error))) {
                free(host_source);
                free(host_result);
                tensor_buffer_free(source);
                tensor_buffer_free(destination);
                return fail("timed upload failed", error);
            }
        }
        if (!tensor_device_synchronize(error, sizeof(error))) {
            free(host_source);
            free(host_result);
            tensor_buffer_free(source);
            tensor_buffer_free(destination);
            return fail("timed upload synchronization failed", error);
        }
        const double upload_elapsed = now_seconds() - start;

        start = now_seconds();
        for (size_t i = 0; i < TEST_ROUNDS; ++i) {
            const size_t offset = i * TEST_BYTES;
            if (!tensor_buffer_copy_device(destination, offset, source, offset, TEST_BYTES,
                                           error, sizeof(error))) {
                free(host_source);
                free(host_result);
                tensor_buffer_free(source);
                tensor_buffer_free(destination);
                return fail("timed device copy failed", error);
            }
        }
        if (!tensor_device_synchronize(error, sizeof(error))) {
            free(host_source);
            free(host_result);
            tensor_buffer_free(source);
            tensor_buffer_free(destination);
            return fail("timed device-copy synchronization failed", error);
        }
        const double copy_elapsed = now_seconds() - start;
        if (sample > 0) {
            upload_samples[sample - 1] = upload_elapsed;
            copy_samples[sample - 1] = copy_elapsed;
        }
    }
    const double upload_seconds = median_seconds(upload_samples, PERFORMANCE_SAMPLES);
    const double copy_seconds = median_seconds(copy_samples, PERFORMANCE_SAMPLES);
    double upload_minimum = 0.0, upload_maximum = 0.0;
    double copy_minimum = 0.0, copy_maximum = 0.0;
    sample_range(upload_samples, PERFORMANCE_SAMPLES, &upload_minimum, &upload_maximum);
    sample_range(copy_samples, PERFORMANCE_SAMPLES, &copy_minimum, &copy_maximum);

    printf("GPU transfer baseline: upload %.2f GB/s, device-copy %.2f GB/s "
           "median (%u samples; upload %.3f-%.3f ms, copy %.3f-%.3f ms; "
           "%u rounds x %u MiB)\n",
           upload_seconds > 0.0 ?
               (double)TEST_BYTES * TEST_ROUNDS / upload_seconds / 1.0e9 : 0.0,
           copy_seconds > 0.0 ?
               (double)TEST_BYTES * TEST_ROUNDS / copy_seconds / 1.0e9 : 0.0,
           PERFORMANCE_SAMPLES, upload_minimum * 1000.0, upload_maximum * 1000.0,
           copy_minimum * 1000.0, copy_maximum * 1000.0,
           TEST_ROUNDS, (unsigned)(TEST_BYTES / (1024u * 1024u)));

    free(host_source);
    free(host_result);
    tensor_buffer_free(source);
    tensor_buffer_free(destination);
    return 0;
}
