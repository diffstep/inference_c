#include "tensor_f16.h"

#include <string.h>

static uint32_t float_bits(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static float bits_float(uint32_t bits) {
    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

float tensor_f16_to_f32(uint16_t value) {
    const uint32_t sign = (uint32_t)(value & 0x8000u) << 16;
    uint32_t exponent = (value >> 10) & 0x1fu;
    uint32_t fraction = value & 0x03ffu;
    uint32_t bits;
    if (exponent == 0) {
        if (fraction == 0) {
            bits = sign;
        } else {
            int adjusted_exponent = -14;
            while ((fraction & 0x0400u) == 0) {
                fraction <<= 1;
                adjusted_exponent--;
            }
            fraction &= 0x03ffu;
            bits = sign | ((uint32_t)(adjusted_exponent + 127) << 23) |
                   (fraction << 13);
        }
    } else if (exponent == 0x1fu) {
        bits = sign | 0x7f800000u | (fraction << 13);
    } else {
        exponent = exponent - 15 + 127;
        bits = sign | (exponent << 23) | (fraction << 13);
    }
    return bits_float(bits);
}

float tensor_bf16_to_f32(uint16_t value) {
    return bits_float((uint32_t)value << 16);
}

uint16_t tensor_f32_to_f16(float value) {
    const uint32_t bits = float_bits(value);
    const uint16_t sign = (uint16_t)((bits >> 16) & 0x8000u);
    const uint32_t exponent_bits = (bits >> 23) & 0xffu;
    const uint32_t fraction = bits & 0x007fffffu;
    if (exponent_bits == 0xffu) {
        if (fraction == 0) return (uint16_t)(sign | 0x7c00u);
        uint16_t payload = (uint16_t)(fraction >> 13);
        if (payload == 0) payload = 1;
        return (uint16_t)(sign | 0x7c00u | payload);
    }
    int exponent = (int)exponent_bits - 127 + 15;
    if (exponent >= 31) return (uint16_t)(sign | 0x7c00u);
    if (exponent <= 0) {
        if (exponent < -10) return sign;
        const uint32_t significand = fraction | 0x00800000u;
        const unsigned shift = (unsigned)(14 - exponent);
        uint32_t rounded = significand >> shift;
        const uint32_t remainder = significand & ((1u << shift) - 1u);
        const uint32_t midpoint = 1u << (shift - 1u);
        if (remainder > midpoint ||
            (remainder == midpoint && (rounded & 1u))) rounded++;
        return (uint16_t)(sign | (uint16_t)rounded);
    }
    uint32_t rounded_fraction = fraction >> 13;
    const uint32_t remainder = fraction & 0x1fffu;
    if (remainder > 0x1000u ||
        (remainder == 0x1000u && (rounded_fraction & 1u))) {
        rounded_fraction++;
        if (rounded_fraction == 0x400u) {
            rounded_fraction = 0;
            exponent++;
            if (exponent >= 31) return (uint16_t)(sign | 0x7c00u);
        }
    }
    return (uint16_t)(sign | ((uint16_t)exponent << 10) |
                      (uint16_t)rounded_fraction);
}
