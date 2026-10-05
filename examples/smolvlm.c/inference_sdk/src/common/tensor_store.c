#include "tensor_store.h"
#include "tensor_f16.h"

#include "json.h"
#include "tensor_f16.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    char *path;
    SafeTensors *file;
} StoreShard;

typedef struct {
    const char *name;
    size_t shard_index;
    const SafeTensorInfo *tensor;
} StoreEntry;

struct TensorStore {
    StoreShard *shards;
    size_t shard_count;
    StoreEntry *entries;
    size_t entry_count;
};

static void set_error(char *error, size_t capacity, const char *message) {
    if (error != NULL && capacity > 0) {
        snprintf(error, capacity, "%s", message);
    }
}

static char *copy_string(const char *value) {
    const size_t length = strlen(value);
    char *copy = malloc(length + 1);
    if (copy != NULL) memcpy(copy, value, length + 1);
    return copy;
}

static char *join_path(const char *directory, const char *name) {
    if (name[0] == '/') return copy_string(name);
    const size_t directory_length = strlen(directory);
    const size_t name_length = strlen(name);
    const int separator = directory_length > 0 && directory[directory_length - 1] != '/';
    if (directory_length > SIZE_MAX - name_length - (size_t)separator - 1) return NULL;
    char *path = malloc(directory_length + name_length + (size_t)separator + 1);
    if (path == NULL) return NULL;
    memcpy(path, directory, directory_length);
    size_t offset = directory_length;
    if (separator) path[offset++] = '/';
    memcpy(path + offset, name, name_length + 1);
    return path;
}

static void store_free_partial(TensorStore *store) {
    if (store == NULL) return;
    for (size_t index = 0; index < store->shard_count; index++) {
        safetensors_close(store->shards[index].file);
        free(store->shards[index].path);
    }
    free(store->shards);
    free(store->entries);
    free(store);
}

static int find_or_add_shard(TensorStore *store, const char *path,
                             size_t *result, char *error,
                             size_t error_capacity) {
    for (size_t index = 0; index < store->shard_count; index++) {
        if (strcmp(store->shards[index].path, path) == 0) {
            *result = index;
            return 1;
        }
    }
    char detail[256];
    SafeTensors *file = NULL;
    if (!safetensors_open(path, &file, detail, sizeof(detail))) {
        if (error != NULL && error_capacity > 0) {
            snprintf(error, error_capacity, "%s: %s", path, detail);
        }
        return 0;
    }
    char *owned_path = copy_string(path);
    if (owned_path == NULL) {
        safetensors_close(file);
        set_error(error, error_capacity, "out of memory storing tensor shard path");
        return 0;
    }
    StoreShard *shards = realloc(store->shards,
                                 (store->shard_count + 1) * sizeof(*shards));
    if (shards == NULL) {
        free(owned_path);
        safetensors_close(file);
        set_error(error, error_capacity, "out of memory adding tensor shard");
        return 0;
    }
    store->shards = shards;
    *result = store->shard_count;
    store->shards[store->shard_count++] = (StoreShard){owned_path, file};
    return 1;
}

static int append_entry(TensorStore *store, size_t shard_index,
                        const SafeTensorInfo *tensor, char *error,
                        size_t error_capacity) {
    for (size_t index = 0; index < store->entry_count; index++) {
        if (strcmp(store->entries[index].name, tensor->name) == 0) {
            set_error(error, error_capacity, "duplicate tensor name in store");
            return 0;
        }
    }
    StoreEntry *entries = realloc(store->entries,
                                  (store->entry_count + 1) * sizeof(*entries));
    if (entries == NULL) {
        set_error(error, error_capacity, "out of memory indexing tensor metadata");
        return 0;
    }
    store->entries = entries;
    store->entries[store->entry_count++] = (StoreEntry){
        tensor->name, shard_index, tensor
    };
    return 1;
}

int tensor_store_open_file(const char *safetensors_path, TensorStore **result,
                           char *error, size_t error_capacity) {
    if (error != NULL && error_capacity > 0) error[0] = '\0';
    if (safetensors_path == NULL || result == NULL) {
        set_error(error, error_capacity, "invalid argument");
        return 0;
    }
    *result = NULL;
    TensorStore *store = calloc(1, sizeof(*store));
    if (store == NULL) {
        set_error(error, error_capacity, "out of memory creating tensor store");
        return 0;
    }
    size_t shard_index;
    if (!find_or_add_shard(store, safetensors_path, &shard_index,
                           error, error_capacity)) {
        store_free_partial(store);
        return 0;
    }
    const SafeTensors *file = store->shards[shard_index].file;
    for (size_t index = 0; index < safetensors_count(file); index++) {
        if (!append_entry(store, shard_index, safetensors_at(file, index),
                          error, error_capacity)) {
            store_free_partial(store);
            return 0;
        }
    }
    *result = store;
    return 1;
}

int tensor_store_open_index(const char *index_path, const char *shard_directory,
                            TensorStore **result, char *error,
                            size_t error_capacity) {
    if (error != NULL && error_capacity > 0) error[0] = '\0';
    if (index_path == NULL || shard_directory == NULL || result == NULL) {
        set_error(error, error_capacity, "invalid argument");
        return 0;
    }
    *result = NULL;
    JsonValue *index_json = NULL;
    if (!json_parse_file(index_path, &index_json, error, error_capacity)) return 0;
    const JsonValue *weight_map = json_object_get(index_json, "weight_map");
    const size_t tensor_count = json_object_size(weight_map);
    if (tensor_count == 0) {
        json_free(index_json);
        set_error(error, error_capacity, "index has no weight_map entries");
        return 0;
    }
    TensorStore *store = calloc(1, sizeof(*store));
    if (store == NULL) {
        json_free(index_json);
        set_error(error, error_capacity, "out of memory creating tensor store");
        return 0;
    }
    int ok = 1;
    for (size_t index = 0; ok && index < tensor_count; index++) {
        const char *tensor_name = json_object_key_at(weight_map, index);
        const char *shard_name = json_string(json_object_value_at(weight_map, index));
        if (tensor_name == NULL || shard_name == NULL || shard_name[0] == '\0') {
            set_error(error, error_capacity, "index weight_map entry is not a shard name");
            ok = 0;
            break;
        }
        char *shard_path = join_path(shard_directory, shard_name);
        size_t shard_index;
        if (shard_path == NULL || !find_or_add_shard(
                store, shard_path, &shard_index, error, error_capacity)) {
            free(shard_path);
            if (error != NULL && error_capacity > 0 && error[0] == '\0')
                set_error(error, error_capacity, "cannot resolve shard path");
            ok = 0;
            break;
        }
        free(shard_path);
        const SafeTensorInfo *tensor = safetensors_find(
            store->shards[shard_index].file, tensor_name);
        if (tensor == NULL) {
            if (error != NULL && error_capacity > 0)
                snprintf(error, error_capacity, "tensor %s missing from shard %s",
                         tensor_name, shard_name);
            ok = 0;
            break;
        }
        ok = append_entry(store, shard_index, tensor, error, error_capacity);
    }
    json_free(index_json);
    if (!ok) {
        store_free_partial(store);
        return 0;
    }
    *result = store;
    return 1;
}

void tensor_store_close(TensorStore *store) {
    store_free_partial(store);
}

size_t tensor_store_count(const TensorStore *store) {
    return store == NULL ? 0 : store->entry_count;
}

const SafeTensorInfo *tensor_store_find(const TensorStore *store,
                                        const char *tensor_name) {
    if (store == NULL || tensor_name == NULL) return NULL;
    for (size_t index = 0; index < store->entry_count; index++) {
        if (strcmp(store->entries[index].name, tensor_name) == 0)
            return store->entries[index].tensor;
    }
    return NULL;
}

int tensor_store_read_raw(const TensorStore *store, const char *tensor_name,
                          void *destination, size_t capacity,
                          char *error, size_t error_capacity) {
    const SafeTensorInfo *tensor = tensor_store_find(store, tensor_name);
    if (tensor == NULL) {
        set_error(error, error_capacity, "tensor not found in store");
        return 0;
    }
    for (size_t index = 0; index < store->entry_count; index++) {
        if (store->entries[index].tensor == tensor) {
            return safetensors_read_raw(
                store->shards[store->entries[index].shard_index].file,
                tensor, destination, capacity, error, error_capacity);
        }
    }
    set_error(error, error_capacity, "tensor metadata is not indexed");
    return 0;
}

static int tensor_store_read_range(const TensorStore *store,
                                   const SafeTensorInfo *tensor,
                                   uint64_t byte_offset, void *destination,
                                   size_t byte_count, char *error,
                                   size_t error_capacity) {
    for (size_t index = 0; index < store->entry_count; index++) {
        if (store->entries[index].tensor == tensor) {
            return safetensors_read_raw_range(
                store->shards[store->entries[index].shard_index].file,
                tensor, byte_offset, destination, byte_count,
                error, error_capacity);
        }
    }
    set_error(error, error_capacity, "tensor metadata is not indexed");
    return 0;
}

int tensor_store_read_f16(const TensorStore *store, const char *tensor_name,
                          uint16_t *destination, size_t capacity,
                          char *error, size_t error_capacity) {
    const SafeTensorInfo *tensor = tensor_store_find(store, tensor_name);
    if (tensor == NULL || destination == NULL) {
        set_error(error, error_capacity, "invalid FP16 tensor read arguments");
        return 0;
    }
    if (tensor->dtype == TENSOR_DTYPE_F16) {
        if (tensor->byte_length > capacity) {
            set_error(error, error_capacity, "FP16 destination is too small");
            return 0;
        }
        return tensor_store_read_raw(store, tensor_name, destination, capacity,
                                     error, error_capacity);
    }
    if (tensor->dtype != TENSOR_DTYPE_F32 && tensor->dtype != TENSOR_DTYPE_BF16) {
        set_error(error, error_capacity, "FP16 load supports F16, BF16, or F32 source tensors");
        return 0;
    }
    const size_t source_element_size = tensor->dtype == TENSOR_DTYPE_F32 ? sizeof(float) : sizeof(uint16_t);
    if (tensor->byte_length % source_element_size != 0) {
        set_error(error, error_capacity, "invalid FP16 source tensor size");
        return 0;
    }
    const uint64_t element_count = tensor->byte_length / source_element_size;
    if (element_count > SIZE_MAX / sizeof(uint16_t) ||
        (size_t)element_count * sizeof(uint16_t) > capacity) {
        set_error(error, error_capacity, "FP16 destination is too small");
        return 0;
    }
    enum { CHUNK_ELEMENTS = 4096 };
    float chunk[CHUNK_ELEMENTS];
    uint16_t half_chunk[CHUNK_ELEMENTS];
    uint64_t offset = 0;
    while (offset < element_count) {
        const size_t count = (size_t)((element_count - offset) < CHUNK_ELEMENTS ?
            (element_count - offset) : CHUNK_ELEMENTS);
        if (tensor->dtype == TENSOR_DTYPE_F32) {
            const size_t bytes = count * sizeof(float);
            if (!tensor_store_read_range(store, tensor, offset * sizeof(float),
                                         chunk, bytes, error, error_capacity)) return 0;
            for (size_t index = 0; index < count; index++)
                destination[(size_t)offset + index] = tensor_f32_to_f16(chunk[index]);
        } else {
            const size_t bytes = count * sizeof(uint16_t);
            if (!tensor_store_read_range(store, tensor, offset * sizeof(uint16_t),
                                         half_chunk, bytes, error, error_capacity)) return 0;
            for (size_t index = 0; index < count; index++)
                destination[(size_t)offset + index] = tensor_f32_to_f16(
                    tensor_bf16_to_f32(half_chunk[index]));
        }
        offset += count;
    }
    return 1;
}

int tensor_store_read_f32(const TensorStore *store, const char *tensor_name,
                          float *destination, size_t capacity,
                          char *error, size_t error_capacity) {
    const SafeTensorInfo *tensor = tensor_store_find(store, tensor_name);
    if (tensor == NULL || destination == NULL) {
        set_error(error, error_capacity, "invalid FP32 tensor read arguments");
        return 0;
    }
    if (tensor->dtype == TENSOR_DTYPE_F32) {
        if (tensor->byte_length > capacity) {
            set_error(error, error_capacity, "FP32 destination is too small");
            return 0;
        }
        return tensor_store_read_raw(store, tensor_name, destination, capacity,
                                     error, error_capacity);
    }
    if ((tensor->dtype != TENSOR_DTYPE_F16 && tensor->dtype != TENSOR_DTYPE_BF16) ||
        tensor->byte_length % sizeof(uint16_t) != 0) {
        set_error(error, error_capacity, "FP32 load supports F16, BF16, or F32 source tensors");
        return 0;
    }
    const uint64_t element_count = tensor->byte_length / sizeof(uint16_t);
    if (element_count > SIZE_MAX / sizeof(float) ||
        (size_t)element_count * sizeof(float) > capacity) {
        set_error(error, error_capacity, "FP32 destination is too small");
        return 0;
    }
    enum { CHUNK_ELEMENTS = 4096 };
    uint16_t chunk[CHUNK_ELEMENTS];
    uint64_t offset = 0;
    while (offset < element_count) {
        const size_t count = (size_t)((element_count - offset) < CHUNK_ELEMENTS ?
            (element_count - offset) : CHUNK_ELEMENTS);
        if (!tensor_store_read_range(store, tensor, offset * sizeof(uint16_t),
                                     chunk, count * sizeof(uint16_t), error, error_capacity))
            return 0;
        for (size_t index = 0; index < count; index++)
            destination[(size_t)offset + index] = tensor->dtype == TENSOR_DTYPE_F16 ?
                tensor_f16_to_f32(chunk[index]) : tensor_bf16_to_f32(chunk[index]);
        offset += count;
    }
    return 1;
}
