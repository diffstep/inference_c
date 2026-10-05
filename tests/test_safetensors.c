#include "safetensors.h"

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void write_u64_le(FILE *file, uint64_t value) {
    unsigned char bytes[8];
    for (size_t index = 0; index < sizeof(bytes); index++) {
        bytes[index] = (unsigned char)(value >> (index * 8));
    }
    assert(fwrite(bytes, 1, sizeof(bytes), file) == sizeof(bytes));
}

static void write_test_file(const char *path) {
    const char *json =
        "{\"__metadata__\":{\"format\":\"pt\"},"
        "\"weight\":{\"dtype\":\"F32\",\"shape\":[2,3],"
        "\"data_offsets\":[0,24]},"
        "\"norm\":{\"dtype\":\"BF16\",\"shape\":[2],"
        "\"data_offsets\":[24,28]}}";
    const size_t json_length = strlen(json);
    const size_t header_length = (json_length + 7) & ~(size_t)7;
    FILE *file = fopen(path, "wb");
    assert(file != NULL);
    write_u64_le(file, header_length);
    assert(fwrite(json, 1, json_length, file) == json_length);
    for (size_t index = json_length; index < header_length; index++) {
        assert(fputc(' ', file) != EOF);
    }
    const float values[6] = {1.0f, -2.0f, 3.5f, 4.0f, 5.0f, 6.0f};
    assert(fwrite(values, sizeof(values[0]), 6, file) == 6);
    const unsigned char norm[4] = {0x80, 0x3f, 0x00, 0x40};
    assert(fwrite(norm, 1, sizeof(norm), file) == sizeof(norm));
    assert(fclose(file) == 0);
}

int main(void) {
    const char *path = "build/safetensors-test.bin";
    write_test_file(path);

    SafeTensors *file = NULL;
    char error[256];
    assert(safetensors_open(path, &file, error, sizeof(error)));
    assert(safetensors_count(file) == 2);

    const SafeTensorInfo *weight = safetensors_find(file, "weight");
    assert(weight != NULL);
    assert(weight->dtype == TENSOR_DTYPE_F32);
    assert(weight->rank == 2);
    assert(weight->shape[0] == 2 && weight->shape[1] == 3);
    assert(weight->byte_length == sizeof(float) * 6);
    unsigned char too_small[8];
    assert(!safetensors_read_raw(file, weight, too_small, sizeof(too_small),
                                 error, sizeof(error)));

    float values[6] = {0};
    assert(safetensors_read_raw(file, weight, values, sizeof(values), error, sizeof(error)));
    assert(values[0] == 1.0f && values[1] == -2.0f && values[2] == 3.5f);
    assert(values[3] == 4.0f && values[4] == 5.0f && values[5] == 6.0f);

    const SafeTensorInfo *norm = safetensors_at(file, 1);
    assert(norm != NULL && norm->dtype == TENSOR_DTYPE_BF16);
    assert(tensor_dtype_size(norm->dtype) == 2);
    safetensors_close(file);
    assert(remove(path) == 0);
    puts("safetensors tests passed");
    return 0;
}
