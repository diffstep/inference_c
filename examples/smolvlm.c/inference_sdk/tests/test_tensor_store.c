#include "tensor_buffer.h"

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void write_u64_le(FILE *file, uint64_t value) {
    unsigned char bytes[8];
    for (size_t index = 0; index < sizeof(bytes); index++)
        bytes[index] = (unsigned char)(value >> (index * 8));
    assert(fwrite(bytes, 1, sizeof(bytes), file) == sizeof(bytes));
}

static void write_tensor_file(const char *path, const char *name, float value) {
    char json[256];
    const int length = snprintf(
        json, sizeof(json),
        "{\"%s\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]}}",
        name);
    assert(length > 0 && (size_t)length < sizeof(json));
    const size_t header_length = ((size_t)length + 7) & ~(size_t)7;
    FILE *file = fopen(path, "wb");
    assert(file != NULL);
    write_u64_le(file, header_length);
    assert(fwrite(json, 1, (size_t)length, file) == (size_t)length);
    for (size_t index = (size_t)length; index < header_length; index++)
        assert(fputc(' ', file) != EOF);
    assert(fwrite(&value, sizeof(value), 1, file) == 1);
    assert(fclose(file) == 0);
}

int main(void) {
    const char *first_path = "build/model-00001-of-00002.safetensors";
    const char *second_path = "build/model-00002-of-00002.safetensors";
    const char *index_path = "build/model.safetensors.index.json";
    write_tensor_file(first_path, "first.weight", 1.25f);
    write_tensor_file(second_path, "second.weight", -2.5f);
    FILE *index = fopen(index_path, "wb");
    assert(index != NULL);
    const char *index_json =
        "{\"metadata\":{\"total_size\":8},\"weight_map\":{" 
        "\"first.weight\":\"model-00001-of-00002.safetensors\","
        "\"second.weight\":\"model-00002-of-00002.safetensors\"}}";
    assert(fwrite(index_json, 1, strlen(index_json), index) == strlen(index_json));
    assert(fclose(index) == 0);

    TensorStore *store = NULL;
    char error[256];
    assert(tensor_store_open_index(index_path, "build", &store,
                                   error, sizeof(error)));
    assert(tensor_store_count(store) == 2);
    const SafeTensorInfo *first = tensor_store_find(store, "first.weight");
    const SafeTensorInfo *second = tensor_store_find(store, "second.weight");
    assert(first != NULL && first->dtype == TENSOR_DTYPE_F32);
    assert(second != NULL && second->shape[0] == 1);
    float first_value = 0.0f;
    float second_value = 0.0f;
    assert(tensor_store_read_raw(store, "first.weight", &first_value,
                                 sizeof(first_value), error, sizeof(error)));
    assert(tensor_store_read_raw(store, "second.weight", &second_value,
                                 sizeof(second_value), error, sizeof(error)));
    assert(first_value == 1.25f && second_value == -2.5f);
    TensorBuffer *loaded = NULL;
    assert(tensor_buffer_load(store, "first.weight", &loaded,
                              error, sizeof(error)));
    assert(tensor_buffer_dtype(loaded) == TENSOR_DTYPE_F32);
    assert(tensor_buffer_element_count(loaded) == 1);
    assert(*(const float *)tensor_buffer_data(loaded) == 1.25f);
    tensor_buffer_free(loaded);
    assert(!tensor_store_read_raw(store, "missing.weight", &first_value,
                                  sizeof(first_value), error, sizeof(error)));
    tensor_store_close(store);

    assert(tensor_store_open_file(first_path, &store, error, sizeof(error)));
    assert(tensor_store_count(store) == 1);
    assert(tensor_store_find(store, "first.weight") != NULL);
    tensor_store_close(store);

    assert(remove(first_path) == 0);
    assert(remove(second_path) == 0);
    assert(remove(index_path) == 0);
    puts("tensor store tests passed");
    return 0;
}
