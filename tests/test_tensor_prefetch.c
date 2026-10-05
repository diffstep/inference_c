#include "tensor_prefetch.h"

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static void write_u64_le(FILE *file, uint64_t value) {
    unsigned char bytes[8];
    for (size_t index = 0; index < sizeof(bytes); index++)
        bytes[index] = (unsigned char)(value >> (index * 8));
    assert(fwrite(bytes, 1, sizeof(bytes), file) == sizeof(bytes));
}

static void write_tensor_file(const char *path) {
    const char *json = "{\"weight\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[0,8]}}";
    const size_t json_length = strlen(json);
    const size_t header_length = (json_length + 7) & ~(size_t)7;
    const float values[] = {1.25f, -3.5f};
    FILE *file = fopen(path, "wb");
    assert(file != NULL);
    write_u64_le(file, header_length);
    assert(fwrite(json, 1, json_length, file) == json_length);
    for (size_t index = json_length; index < header_length; index++)
        assert(fputc(' ', file) != EOF);
    assert(fwrite(values, sizeof(values), 1, file) == 1);
    assert(fclose(file) == 0);
}

int main(void) {
    const char *path = "build/prefetch-model.safetensors";
    write_tensor_file(path);
    TensorStore *store = NULL;
    TensorPrefetch *prefetch = NULL;
    TensorBuffer *loaded = NULL;
    char error[256] = {0};
    assert(tensor_store_open_file(path, &store, error, sizeof(error)));
    assert(tensor_prefetch_create(store, &prefetch, error, sizeof(error)));

    assert(tensor_prefetch_request(prefetch, "weight", error, sizeof(error)));
    assert(!tensor_prefetch_request(prefetch, "weight", error, sizeof(error)));
    assert(tensor_prefetch_wait(prefetch, &loaded, error, sizeof(error)));
    assert(tensor_buffer_element_count(loaded) == 2);
    const float *values = tensor_buffer_data(loaded);
    assert(values[0] == 1.25f && values[1] == -3.5f);
    tensor_buffer_free(loaded);
    loaded = NULL;

    assert(tensor_prefetch_request(prefetch, "missing", error, sizeof(error)));
    assert(!tensor_prefetch_wait(prefetch, &loaded, error, sizeof(error)));
    assert(strstr(error, "not found") != NULL);

    tensor_prefetch_free(prefetch);
    tensor_store_close(store);
    assert(remove(path) == 0);
    puts("tensor prefetch tests passed");
    return 0;
}
