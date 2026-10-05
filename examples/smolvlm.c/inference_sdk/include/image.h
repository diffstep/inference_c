#ifndef IMAGE_H
#define IMAGE_H

#include <stddef.h>

typedef struct {
    size_t width;
    size_t height;
    unsigned char *rgb;
} RgbImage;

typedef struct {
    size_t count;
    size_t rows;
    size_t columns;
    float **normalized_chw;
} ImageTileSet;

typedef unsigned char *(*ImageAllocate)(size_t size);

int image_decode_rgb_rust(const char *path, ImageAllocate allocate,
                           unsigned char **pixels, size_t *width,
                           size_t *height);

int image_load(const char *path, RgbImage *result, char *error,
               size_t error_capacity);
void image_free(RgbImage *image);
int image_resize_normalize(const RgbImage *image, size_t width, size_t height,
                           const float mean[3], const float standard_deviation[3],
                           float **result, char *error, size_t error_capacity);
int image_preprocess_tiles(const RgbImage *image, size_t encoder_edge,
                           size_t longest_edge, const float mean[3],
                           const float standard_deviation[3], ImageTileSet *result,
                           char *error, size_t error_capacity);
void image_tile_set_free(ImageTileSet *tiles);

#endif
