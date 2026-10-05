#ifndef TENSOR_F16_H
#define TENSOR_F16_H

#include <stdint.h>

uint16_t tensor_f32_to_f16(float value);
float tensor_f16_to_f32(uint16_t value);
float tensor_bf16_to_f32(uint16_t value);

#endif
