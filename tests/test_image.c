#include "image.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    const char *path = "build/test-image.ppm";
    FILE *file = fopen(path, "wb");
    assert(file != NULL);
    const unsigned char pixels[] = {255, 0, 0, 0, 255, 0};
    fputs("P6\n2 1\n255\n", file);
    assert(fwrite(pixels, 1, sizeof(pixels), file) == sizeof(pixels));
    fclose(file);

    RgbImage image;
    char error[256] = {0};
    assert(image_load(path, &image, error, sizeof(error)));
    assert(image.width == 2 && image.height == 1);
    const float mean[] = {0.5f, 0.5f, 0.5f};
    const float stddev[] = {0.5f, 0.5f, 0.5f};
    float *normalized = NULL;
    assert(image_resize_normalize(&image, 2, 1, mean, stddev, &normalized,
                                  error, sizeof(error)));
    assert(normalized[0] == 1.0f && normalized[1] == -1.0f);
    assert(normalized[2] == -1.0f && normalized[3] == 1.0f);
    assert(normalized[4] == -1.0f && normalized[5] == -1.0f);
    free(normalized);
    ImageTileSet tiles;
    assert(image_preprocess_tiles(&image, 2, 4, mean, stddev, &tiles,
                                  error, sizeof(error)));
    assert(tiles.rows == 1 && tiles.columns == 2 && tiles.count == 3);
    assert(tiles.normalized_chw[0] != NULL && tiles.normalized_chw[1] != NULL &&
           tiles.normalized_chw[2] != NULL);
    image_tile_set_free(&tiles);
    image_free(&image);
    remove(path);
    puts("image tests passed");
    return 0;
}
