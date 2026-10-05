#include "image.h"

#include <inttypes.h>
#include <stdio.h>

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <image-path>\n", argv[0]);
        return 2;
    }
    RgbImage image = {0};
    char error[256] = {0};
    if (!image_load(argv[1], &image, error, sizeof(error))) {
        fprintf(stderr, "%s\n", error);
        return 1;
    }
    uint64_t hash = UINT64_C(1469598103934665603);
    const size_t size = image.width * image.height * 3;
    for (size_t index = 0; index < size; index++) {
        hash ^= image.rgb[index];
        hash *= UINT64_C(1099511628211);
    }
    printf("decoded_rgb_hash=%016" PRIx64 " %zux%zu\n", hash,
           image.width, image.height);
    image_free(&image);
    return 0;
}
