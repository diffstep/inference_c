#include "image.h"
#include "timing.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifndef IMAGE_DECODER_RUST
#define STB_IMAGE_IMPLEMENTATION
#include "../../third_party/stb_image.h"
#endif

static void set_error(char *error, size_t capacity, const char *message) {
    if (error != NULL && capacity > 0) snprintf(error, capacity, "%s", message);
}

#ifdef IMAGE_DECODER_RUST
static unsigned char *image_allocate(size_t size) {
    return malloc(size);
}
#endif

int image_load(const char *path, RgbImage *result, char *error,
               size_t error_capacity) {
    if (result != NULL) memset(result, 0, sizeof(*result));
    if (path == NULL || result == NULL) {
        set_error(error, error_capacity, "invalid image arguments");
        return 0;
    }
#ifdef IMAGE_DECODER_RUST
    unsigned char *pixels = NULL;
    if (!image_decode_rgb_rust(path, image_allocate, &pixels, &result->width,
                               &result->height)) {
        set_error(error, error_capacity, "Rust image decoder failed");
        return 0;
    }
    result->rgb = pixels;
#else
    int width = 0, height = 0, channels = 0;
    unsigned char *pixels = stbi_load(path, &width, &height, &channels, 3);
    if (pixels == NULL || width <= 0 || height <= 0) {
        if (pixels != NULL) stbi_image_free(pixels);
        if (error != NULL && error_capacity > 0)
            snprintf(error, error_capacity, "cannot decode image: %s", stbi_failure_reason());
        return 0;
    }
    result->width = (size_t)width;
    result->height = (size_t)height;
    result->rgb = pixels;
#endif
    return 1;
}

void image_free(RgbImage *image) {
    if (image == NULL) return;
#ifdef IMAGE_DECODER_RUST
    free(image->rgb);
#else
    stbi_image_free(image->rgb);
#endif
    memset(image, 0, sizeof(*image));
}

static float sinc(float value) {
    if (fabsf(value) < 1e-6f) return 1.0f;
    const float angle = 3.14159265358979323846f * value;
    return sinf(angle) / angle;
}

typedef struct {
    size_t *offsets;
    size_t *indices;
    float *weights;
} ResizeAxis;

static void resize_axis_free(ResizeAxis *axis) {
    free(axis->offsets);
    free(axis->indices);
    free(axis->weights);
    memset(axis, 0, sizeof(*axis));
}

static void resize_axis_bounds(size_t source_size, size_t output_index,
                               size_t output_size, float *scale, float *filter,
                               float *center, int *first, int *last) {
    *scale = (float)source_size / (float)output_size;
    *filter = *scale > 1.0f ? *scale : 1.0f;
    *center = ((float)output_index + 0.5f) * *scale - 0.5f;
    const float radius = 3.0f * *filter;
    *first = (int)ceilf(*center - radius);
    *last = (int)floorf(*center + radius);
}

static int resize_axis_init(ResizeAxis *axis, size_t source_size,
                            size_t output_size) {
    memset(axis, 0, sizeof(*axis));
    if (source_size == 0 || output_size == 0 || output_size == SIZE_MAX ||
        output_size + 1 > SIZE_MAX / sizeof(*axis->offsets)) return 0;
    axis->offsets = malloc((output_size + 1) * sizeof(*axis->offsets));
    if (axis->offsets == NULL) return 0;
    size_t tap_count = 0;
    axis->offsets[0] = 0;
    for (size_t output = 0; output < output_size; output++) {
        float scale, filter, center;
        int first, last;
        resize_axis_bounds(source_size, output, output_size, &scale, &filter,
                           &center, &first, &last);
        for (int sample = first; sample <= last; sample++) {
            const float distance = ((float)sample - center) / filter;
            if (fabsf(distance) < 3.0f) {
                if (tap_count == SIZE_MAX) return 0;
                tap_count++;
            }
        }
        axis->offsets[output + 1] = tap_count;
    }
    if (tap_count > SIZE_MAX / sizeof(*axis->indices) ||
        tap_count > SIZE_MAX / sizeof(*axis->weights)) return 0;
    axis->indices = malloc(tap_count * sizeof(*axis->indices));
    axis->weights = malloc(tap_count * sizeof(*axis->weights));
    if (axis->indices == NULL || axis->weights == NULL) return 0;
    size_t tap = 0;
    for (size_t output = 0; output < output_size; output++) {
        float scale, filter, center;
        int first, last;
        resize_axis_bounds(source_size, output, output_size, &scale, &filter,
                           &center, &first, &last);
        for (int sample = first; sample <= last; sample++) {
            const float distance = ((float)sample - center) / filter;
            if (fabsf(distance) >= 3.0f) continue;
            axis->indices[tap] = (size_t)(sample < 0 ? 0 :
                sample >= (int)source_size ? (int)source_size - 1 : sample);
            axis->weights[tap] = sinc(distance) * sinc(distance / 3.0f);
            tap++;
        }
    }
    return 1;
}

static int resize_lanczos3(const RgbImage *source, size_t width, size_t height,
                           RgbImage *result, char *error, size_t capacity) {
    if (source->width == width && source->height == height) {
        result->rgb = malloc(width * height * 3);
        if (result->rgb == NULL) {
            set_error(error, capacity, "out of memory copying image");
            return 0;
        }
        memcpy(result->rgb, source->rgb, width * height * 3);
        result->width = width;
        result->height = height;
        return 1;
    }
    ResizeAxis x_axis = {0}, y_axis = {0};
    if (!resize_axis_init(&x_axis, source->width, width) ||
        !resize_axis_init(&y_axis, source->height, height)) {
        resize_axis_free(&x_axis);
        resize_axis_free(&y_axis);
        set_error(error, capacity, "out of memory preparing resize filters");
        return 0;
    }
    if (width == 0 || height == 0 || width > SIZE_MAX / source->height / 3 ||
        width * source->height * 3 > SIZE_MAX / sizeof(float) ||
        width > SIZE_MAX / height / 3) {
        resize_axis_free(&x_axis);
        resize_axis_free(&y_axis);
        set_error(error, capacity, "resized image dimensions overflow");
        return 0;
    }
    float *horizontal = malloc(width * source->height * 3 * sizeof(float));
    unsigned char *pixels = malloc(width * height * 3);
    if (horizontal == NULL || pixels == NULL) {
        free(horizontal);
        free(pixels);
        resize_axis_free(&x_axis);
        resize_axis_free(&y_axis);
        set_error(error, capacity, "out of memory resizing image");
        return 0;
    }
    for (size_t y = 0; y < source->height; y++) for (size_t x = 0; x < width; x++) {
        const size_t tap_begin = x_axis.offsets[x];
        const size_t tap_end = x_axis.offsets[x + 1];
        float weight_sum = 0.0f;
        for (size_t tap = tap_begin; tap < tap_end; tap++)
            weight_sum += x_axis.weights[tap];
        for (size_t channel = 0; channel < 3; channel++) {
            float sum = 0.0f;
            for (size_t tap = tap_begin; tap < tap_end; tap++)
                sum += source->rgb[(y * source->width + x_axis.indices[tap]) * 3 + channel] *
                       x_axis.weights[tap];
            horizontal[(y * width + x) * 3 + channel] = sum / weight_sum;
        }
    }
    for (size_t y = 0; y < height; y++) for (size_t x = 0; x < width; x++) {
        const size_t tap_begin = y_axis.offsets[y];
        const size_t tap_end = y_axis.offsets[y + 1];
        float weight_sum = 0.0f;
        for (size_t tap = tap_begin; tap < tap_end; tap++)
            weight_sum += y_axis.weights[tap];
        for (size_t channel = 0; channel < 3; channel++) {
            float sum = 0.0f;
            for (size_t tap = tap_begin; tap < tap_end; tap++)
                sum += horizontal[(y_axis.indices[tap] * width + x) * 3 + channel] *
                       y_axis.weights[tap];
            float value = sum / weight_sum;
            if (value < 0.0f) value = 0.0f;
            if (value > 255.0f) value = 255.0f;
            pixels[(y * width + x) * 3 + channel] = (unsigned char)lrintf(value);
        }
    }
    free(horizontal);
    resize_axis_free(&x_axis);
    resize_axis_free(&y_axis);
    result->width = width;
    result->height = height;
    result->rgb = pixels;
    return 1;
}

static size_t ceil_multiple(size_t value, size_t multiple) {
    return value / multiple + (value % multiple != 0);
}

void image_tile_set_free(ImageTileSet *tiles) {
    if (tiles == NULL) return;
    for (size_t i = 0; i < tiles->count; i++) free(tiles->normalized_chw[i]);
    free(tiles->normalized_chw);
    memset(tiles, 0, sizeof(*tiles));
}

int image_preprocess_tiles(const RgbImage *source, size_t edge, size_t longest_edge,
                           const float mean[3], const float standard_deviation[3],
                           ImageTileSet *result, char *error, size_t capacity) {
    if (result != NULL) memset(result, 0, sizeof(*result));
    if (source == NULL || source->rgb == NULL || edge == 0 || longest_edge == 0 ||
        mean == NULL || standard_deviation == NULL || result == NULL) {
        set_error(error, capacity, "invalid image tiling arguments");
        return 0;
    }
    size_t resized_width, resized_height;
    if (source->width >= source->height) {
        resized_width = longest_edge;
        resized_height = (longest_edge * source->height / source->width);
        if (resized_height == 0) resized_height = 1;
        if (resized_height % 2 != 0) resized_height++;
    } else {
        resized_height = longest_edge;
        resized_width = longest_edge * source->width / source->height;
        if (resized_width == 0) resized_width = 1;
        if (resized_width % 2 != 0) resized_width++;
    }
    RgbImage long_resized = {0}, snapped = {0}, global = {0};
    double phase_started = timing_start();
    if (!resize_lanczos3(source, resized_width, resized_height, &long_resized, error, capacity)) {
        timing_report("image_resize_long_edge", phase_started);
        return 0;
    }
    timing_report("image_resize_long_edge", phase_started);
    size_t snapped_width, snapped_height;
    if (resized_width >= resized_height) {
        snapped_width = ceil_multiple(resized_width, edge) * edge;
        const size_t proportional_height = snapped_width * resized_height / resized_width;
        snapped_height = ceil_multiple(proportional_height, edge) * edge;
        if (snapped_height < edge) snapped_height = edge;
    } else {
        snapped_height = ceil_multiple(resized_height, edge) * edge;
        const size_t proportional_width = snapped_height * resized_width / resized_height;
        snapped_width = ceil_multiple(proportional_width, edge) * edge;
        if (snapped_width < edge) snapped_width = edge;
    }
    phase_started = timing_start();
    const int snapped_ok = resize_lanczos3(&long_resized, snapped_width, snapped_height,
                                            &snapped, error, capacity);
    timing_report("image_resize_snapped", phase_started);
    if (!snapped_ok) {
        image_free(&long_resized);
        image_free(&snapped);
        image_free(&global);
        return 0;
    }
    phase_started = timing_start();
    const int global_ok = resize_lanczos3(&snapped, edge, edge, &global, error, capacity);
    timing_report("image_resize_global", phase_started);
    if (!global_ok) {
        image_free(&long_resized);
        image_free(&snapped);
        image_free(&global);
        return 0;
    }
    const size_t rows = snapped_height / edge;
    const size_t columns = snapped_width / edge;
    const int has_crops = rows > 1 || columns > 1;
    const size_t crop_count = has_crops ? rows * columns : 0;
    if (crop_count == SIZE_MAX || crop_count + 1 > SIZE_MAX / sizeof(float *)) {
        image_free(&long_resized); image_free(&snapped); image_free(&global);
        set_error(error, capacity, "too many image crops");
        return 0;
    }
    result->count = crop_count + 1;
    result->rows = has_crops ? rows : 0;
    result->columns = has_crops ? columns : 0;
    result->normalized_chw = calloc(result->count, sizeof(*result->normalized_chw));
    if (result->normalized_chw == NULL) {
        image_free(&long_resized); image_free(&snapped); image_free(&global);
        memset(result, 0, sizeof(*result));
        set_error(error, capacity, "out of memory allocating image crops");
        return 0;
    }
    int ok = 1;
    phase_started = timing_start();
    for (size_t row = 0; ok && row < (has_crops ? rows : 0); row++)
        for (size_t column = 0; ok && column < columns; column++) {
            RgbImage crop = {edge, edge, malloc(edge * edge * 3)};
            if (crop.rgb == NULL) {
                set_error(error, capacity, "out of memory copying image crop");
                ok = 0;
                break;
            }
            for (size_t y = 0; y < edge; y++)
                memcpy(crop.rgb + y * edge * 3,
                       snapped.rgb + ((row * edge + y) * snapped_width + column * edge) * 3,
                       edge * 3);
            ok = image_resize_normalize(&crop, edge, edge, mean, standard_deviation,
                &result->normalized_chw[row * columns + column], error, capacity);
            image_free(&crop);
        }
    timing_report("image_crop_normalize", phase_started);
    phase_started = timing_start();
    if (ok) ok = image_resize_normalize(&global, edge, edge, mean, standard_deviation,
        &result->normalized_chw[crop_count], error, capacity);
    timing_report("image_global_normalize", phase_started);
    image_free(&long_resized); image_free(&snapped); image_free(&global);
    if (!ok) image_tile_set_free(result);
    return ok;
}

int image_resize_normalize(const RgbImage *image, size_t width, size_t height,
                           const float mean[3], const float standard_deviation[3],
                           float **result, char *error, size_t error_capacity) {
    if (result != NULL) *result = NULL;
    if (image == NULL || image->rgb == NULL || image->width == 0 || image->height == 0 ||
        width == 0 || height == 0 || mean == NULL || standard_deviation == NULL ||
        result == NULL || width > SIZE_MAX / height || width * height > SIZE_MAX / (3 * sizeof(float))) {
        set_error(error, error_capacity, "invalid image resize arguments");
        return 0;
    }
    for (size_t channel = 0; channel < 3; channel++) {
        if (standard_deviation[channel] == 0.0f) {
            set_error(error, error_capacity, "image standard deviation cannot be zero");
            return 0;
        }
    }
    const size_t count = width * height * 3;
    float *output = malloc(count * sizeof(*output));
    if (output == NULL) {
        set_error(error, error_capacity, "out of memory resizing image");
        return 0;
    }
    for (size_t y = 0; y < height; y++) {
        const float source_y = ((float)y + 0.5f) * (float)image->height / (float)height - 0.5f;
        const float bounded_y = source_y < 0.0f ? 0.0f : source_y > (float)(image->height - 1) ? (float)(image->height - 1) : source_y;
        const size_t y0 = (size_t)bounded_y;
        const size_t y1 = y0 + 1 < image->height ? y0 + 1 : y0;
        const float fy = bounded_y - (float)y0;
        for (size_t x = 0; x < width; x++) {
            const float source_x = ((float)x + 0.5f) * (float)image->width / (float)width - 0.5f;
            const float bounded_x = source_x < 0.0f ? 0.0f : source_x > (float)(image->width - 1) ? (float)(image->width - 1) : source_x;
            const size_t x0 = (size_t)bounded_x;
            const size_t x1 = x0 + 1 < image->width ? x0 + 1 : x0;
            const float fx = bounded_x - (float)x0;
            for (size_t channel = 0; channel < 3; channel++) {
                const float top = image->rgb[(y0 * image->width + x0) * 3 + channel] * (1.0f - fx) +
                    image->rgb[(y0 * image->width + x1) * 3 + channel] * fx;
                const float bottom = image->rgb[(y1 * image->width + x0) * 3 + channel] * (1.0f - fx) +
                    image->rgb[(y1 * image->width + x1) * 3 + channel] * fx;
                const float value = (top * (1.0f - fy) + bottom * fy) / 255.0f;
                output[channel * width * height + y * width + x] =
                    (value - mean[channel]) / standard_deviation[channel];
            }
        }
    }
    *result = output;
    return 1;
}
