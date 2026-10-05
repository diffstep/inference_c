#ifndef SAFETENSORS_H
#define SAFETENSORS_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    TENSOR_DTYPE_UNKNOWN,
    TENSOR_DTYPE_BOOL,
    TENSOR_DTYPE_U8,
    TENSOR_DTYPE_I8,
    TENSOR_DTYPE_I16,
    TENSOR_DTYPE_U16,
    TENSOR_DTYPE_I32,
    TENSOR_DTYPE_U32,
    TENSOR_DTYPE_I64,
    TENSOR_DTYPE_U64,
    TENSOR_DTYPE_F8_E4M3,
    TENSOR_DTYPE_F8_E5M2,
    TENSOR_DTYPE_F16,
    TENSOR_DTYPE_BF16,
    TENSOR_DTYPE_F32,
    TENSOR_DTYPE_F64
} TensorDType;

typedef struct {
    const char *name;
    TensorDType dtype;
    size_t rank;
    const uint64_t *shape;
    uint64_t data_offset;
    uint64_t byte_length;
} SafeTensorInfo;

typedef struct SafeTensors SafeTensors;

const char *tensor_dtype_name(TensorDType dtype);
size_t tensor_dtype_size(TensorDType dtype);
int safetensors_open(const char *path, SafeTensors **result,
                     char *error, size_t error_capacity);
void safetensors_close(SafeTensors *file);
size_t safetensors_count(const SafeTensors *file);
const SafeTensorInfo *safetensors_at(const SafeTensors *file, size_t index);
const SafeTensorInfo *safetensors_find(const SafeTensors *file, const char *name);
int safetensors_read_raw(const SafeTensors *file, const SafeTensorInfo *tensor,
                         void *destination, size_t capacity,
                         char *error, size_t error_capacity);
int safetensors_read_raw_range(const SafeTensors *file,
                               const SafeTensorInfo *tensor,
                               uint64_t byte_offset, void *destination,
                               size_t byte_count, char *error,
                               size_t error_capacity);

#ifdef __cplusplus
}
#endif

#endif
