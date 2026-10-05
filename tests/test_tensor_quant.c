#include "tensor_quant.h"
#include "tensor_device.h"
#include "tensor_f16.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static uint16_t read_u16(const uint8_t *p) { return (uint16_t)p[0] | ((uint16_t)p[1] << 8); }
static int check_output(const uint8_t *packed, size_t block_bytes, int q8,
        const uint16_t *input, const uint16_t *actual) {
    for (size_t row=0; row<2; ++row) {
        const uint8_t *block=packed+row*block_bytes;
        const float scale=tensor_f16_to_f32(read_u16(block));
        float expected=0.0f;
        for (size_t k=0;k<32;++k) {
            int q;
            if (q8) q=(int)(int8_t)block[2+k];
            else { const uint8_t packed_q=block[2+(k&15)]; q=(int)(k<16 ? packed_q&15 : packed_q>>4)-8; }
            expected+=tensor_f16_to_f32(input[k])*scale*(float)q;
        }
        if (fabsf(tensor_f16_to_f32(actual[row])-expected)>0.02f) return 0;
    }
    return 1;
}

int main(void) {
    uint16_t values[64];
    for (size_t i = 0; i < 64; ++i)
        values[i] = tensor_f32_to_f16((float)((int)(i % 32) - 16) / 8.0f);
    TensorBuffer *q8 = NULL, *q4 = NULL;
    char error[256] = {0};
    if (!tensor_quantize_linear_weight_f16(values, 2, 32, TENSOR_WEIGHT_Q8_0,
                                           &q8, error, sizeof(error))) {
        fprintf(stderr, "Q8 quantization failed: %s\n", error); return 1;
    }
    if (!tensor_quantize_linear_weight_f16(values, 2, 32, TENSOR_WEIGHT_Q4_0,
                                           &q4, error, sizeof(error))) {
        fprintf(stderr, "Q4 quantization failed: %s\n", error); return 1;
    }
    const uint64_t *s8 = tensor_buffer_shape(q8), *s4 = tensor_buffer_shape(q4);
    if (tensor_buffer_dtype(q8) != TENSOR_DTYPE_U8 || s8[0] != 2 || s8[1] != 1 || s8[2] != 34 ||
        tensor_buffer_byte_length(q8) != 68 || s4[0] != 2 || s4[1] != 1 || s4[2] != 18 ||
        tensor_buffer_byte_length(q4) != 36) {
        fprintf(stderr, "unexpected packed Q4/Q8 block layout\n"); return 1;
    }
    const uint8_t *bytes8 = tensor_buffer_data(q8), *bytes4 = tensor_buffer_data(q4);
    if (bytes8[0] == 0 && bytes8[1] == 0) {
        fprintf(stderr, "Q8 scale unexpectedly zero\n"); return 1;
    }
    if (bytes4[0] == 0 && bytes4[1] == 0) {
        fprintf(stderr, "Q4 scale unexpectedly zero\n"); return 1;
    }
    if ((int8_t)bytes8[2] != -127 || (int8_t)bytes8[33] != 119 || bytes4[2] != 0x80) {
        fprintf(stderr,"Q4_0/Q8_0 quantized values do not match expected block ordering: q8=%d,%d q4=%02x\n",
            (int)(int8_t)bytes8[2],(int)(int8_t)bytes8[33],bytes4[2]); return 1;
    }
    if (tensor_device_available()) {
        const uint64_t xshape[2]={1,32}, yshape[2]={1,2};
        TensorBuffer *x=NULL,*w8=NULL,*w4=NULL,*y=NULL;
        uint16_t input[32];
        for (size_t i=0;i<32;++i) input[i]=tensor_f32_to_f16(0.125f*(float)((int)i-12));
        int ok=tensor_buffer_create_device(TENSOR_DTYPE_F16,xshape,2,&x,error,sizeof(error)) &&
            tensor_buffer_upload(x,0,input,sizeof(input),error,sizeof(error)) &&
            tensor_buffer_create_device(TENSOR_DTYPE_U8,s8,3,&w8,error,sizeof(error)) &&
            tensor_buffer_upload(w8,0,bytes8,tensor_buffer_byte_length(q8),error,sizeof(error)) &&
            tensor_buffer_create_device(TENSOR_DTYPE_U8,s4,3,&w4,error,sizeof(error)) &&
            tensor_buffer_upload(w4,0,bytes4,tensor_buffer_byte_length(q4),error,sizeof(error)) &&
            tensor_buffer_create_device(TENSOR_DTYPE_F16,yshape,2,&y,error,sizeof(error));
        uint16_t result[2];
        if (ok) ok=tensor_linear_q8_0_f16(x,w8,y,error,sizeof(error)) &&
            tensor_buffer_download(y,0,result,sizeof(result),error,sizeof(error));
        if (!ok || !check_output(bytes8,34,1,input,result)) { fprintf(stderr,"GPU Q8 linear consistency failed: %s\n",error); return 1; }
        if (!tensor_linear_q4_0_f16(x,w4,y,error,sizeof(error)) ||
            !tensor_buffer_download(y,0,result,sizeof(result),error,sizeof(error))) {
            fprintf(stderr,"GPU Q4 linear failed: %s\n",error); return 1;
        }
        if (!check_output(bytes4,18,0,input,result)) { fprintf(stderr,"GPU Q4 linear consistency failed\n"); return 1; }
        tensor_buffer_free(x);tensor_buffer_free(w8);tensor_buffer_free(w4);tensor_buffer_free(y);
    }
    uint16_t bad[33] = {0};
    TensorBuffer *invalid = NULL;
    if (tensor_quantize_linear_weight_f16(bad, 1, 33, TENSOR_WEIGHT_Q8_0,
                                           &invalid, error, sizeof(error))) {
        fprintf(stderr, "accepted non-block-aligned input width\n"); return 1;
    }
    tensor_buffer_free(q8); tensor_buffer_free(q4); tensor_buffer_free(invalid);
    puts("Q4_0/Q8_0 block layout and validation passed");
    return 0;
}
