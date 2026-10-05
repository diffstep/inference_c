#ifndef VISION_MODEL_H
#define VISION_MODEL_H

#include <stddef.h>

typedef struct VisionModel VisionModel;

int vision_model_open(const char *model_directory, VisionModel **result,
                      char *error, size_t error_capacity);
void vision_model_close(VisionModel *model);
size_t vision_model_input_size(const VisionModel *model);
size_t vision_model_output_tokens(const VisionModel *model);
size_t vision_model_output_width(const VisionModel *model);
int vision_model_uses_fused_sequence(const VisionModel *model);
int vision_model_encode_tile(VisionModel *model, const float *normalized_rgb_chw,
                             float **features, char *error,
                             size_t error_capacity);

#endif
