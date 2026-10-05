#include "paged_kv_cache.h"

#include <pthread.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    uint32_t generation;
    uint32_t *blocks;
    size_t block_count;
    size_t block_capacity;
    size_t token_length;
    int active;
    int decoding;
} PagedSequence;

struct TensorPagedKVCache {
    size_t block_size;
    size_t block_count;
    size_t free_count;
    uint32_t *free_blocks;
    size_t max_sequences;
    PagedSequence *sequences;
    pthread_mutex_t mutex;
    TensorPagedKVBackendStateKind backend_state_kind;
    void *backend_state;
    TensorPagedKVBackendStateDestroy destroy_backend_state;
    TensorPagedKVBackendSequenceRelease release_backend_sequence;
};

static TensorPagedKVSequence make_handle(size_t slot, uint32_t generation) {
    return ((uint64_t)generation << 32) | (uint64_t)(slot + 1);
}

static PagedSequence *find_sequence(TensorPagedKVCache *cache,
                                    TensorPagedKVSequence handle) {
    uint32_t encoded_slot = (uint32_t)handle;
    uint32_t generation = (uint32_t)(handle >> 32);
    if (encoded_slot == 0 || generation == 0) return NULL;
    size_t slot = (size_t)(encoded_slot - 1);
    if (slot >= cache->max_sequences) return NULL;
    PagedSequence *sequence = &cache->sequences[slot];
    return sequence->active && sequence->generation == generation ? sequence : NULL;
}

int tensor_paged_kv_cache_create(size_t block_size, size_t block_count,
                                 size_t max_sequences,
                                 TensorPagedKVCache **out) {
    if (out == NULL || block_size == 0 || block_count == 0 ||
        block_count > UINT32_MAX || max_sequences == 0 ||
        max_sequences > UINT32_MAX ||
        block_count > SIZE_MAX / block_size ||
        block_count > SIZE_MAX / sizeof(uint32_t) ||
        max_sequences > SIZE_MAX / sizeof(PagedSequence)) return 0;
    *out = NULL;
    TensorPagedKVCache *cache = calloc(1, sizeof(*cache));
    if (cache == NULL) return 0;
    cache->free_blocks = malloc(block_count * sizeof(*cache->free_blocks));
    cache->sequences = calloc(max_sequences, sizeof(*cache->sequences));
    if (cache->free_blocks == NULL || cache->sequences == NULL ||
        pthread_mutex_init(&cache->mutex, NULL) != 0) {
        free(cache->free_blocks);
        free(cache->sequences);
        free(cache);
        return 0;
    }
    cache->block_size = block_size;
    cache->block_count = block_count;
    cache->free_count = block_count;
    cache->max_sequences = max_sequences;
    for (size_t i = 0; i < block_count; ++i)
        cache->free_blocks[i] = (uint32_t)(block_count - i - 1);
    *out = cache;
    return 1;
}

void tensor_paged_kv_cache_free(TensorPagedKVCache *cache) {
    if (cache == NULL) return;
    if (cache->destroy_backend_state != NULL && cache->backend_state != NULL)
        cache->destroy_backend_state(cache->backend_state);
    for (size_t i = 0; i < cache->max_sequences; ++i)
        free(cache->sequences[i].blocks);
    pthread_mutex_destroy(&cache->mutex);
    free(cache->sequences);
    free(cache->free_blocks);
    free(cache);
}

int tensor_paged_kv_sequence_create(TensorPagedKVCache *cache,
                                    TensorPagedKVSequence *out) {
    if (cache == NULL || out == NULL) return 0;
    pthread_mutex_lock(&cache->mutex);
    for (size_t i = 0; i < cache->max_sequences; ++i) {
        PagedSequence *sequence = &cache->sequences[i];
        if (!sequence->active) {
            uint32_t generation = sequence->generation + 1;
            if (generation == 0) generation = 1;
            sequence->generation = generation;
            sequence->active = 1;
            sequence->token_length = 0;
            sequence->block_count = 0;
            sequence->decoding = 0;
            *out = make_handle(i, generation);
            pthread_mutex_unlock(&cache->mutex);
            return 1;
        }
    }
    pthread_mutex_unlock(&cache->mutex);
    return 0;
}

int tensor_paged_kv_sequence_reserve(TensorPagedKVCache *cache,
                                     TensorPagedKVSequence handle,
                                     size_t token_capacity) {
    if (cache == NULL || token_capacity > SIZE_MAX - (cache->block_size - 1))
        return 0;
    size_t needed = (token_capacity + cache->block_size - 1) / cache->block_size;
    pthread_mutex_lock(&cache->mutex);
    PagedSequence *sequence = find_sequence(cache, handle);
    if (sequence == NULL) {
        pthread_mutex_unlock(&cache->mutex);
        return 0;
    }
    if (needed <= sequence->block_count) {
        pthread_mutex_unlock(&cache->mutex);
        return 1;
    }
    size_t additional = needed - sequence->block_count;
    if (additional > cache->free_count) {
        pthread_mutex_unlock(&cache->mutex);
        return 0;
    }
    if (needed > sequence->block_capacity) {
        size_t capacity = sequence->block_capacity == 0 ? 4 : sequence->block_capacity;
        while (capacity < needed) {
            if (capacity > SIZE_MAX / 2) { capacity = needed; break; }
            capacity *= 2;
        }
        if (capacity > SIZE_MAX / sizeof(*sequence->blocks)) {
            pthread_mutex_unlock(&cache->mutex);
            return 0;
        }
        uint32_t *blocks = realloc(sequence->blocks, capacity * sizeof(*blocks));
        if (blocks == NULL) {
            pthread_mutex_unlock(&cache->mutex);
            return 0;
        }
        sequence->blocks = blocks;
        sequence->block_capacity = capacity;
    }
    for (size_t i = 0; i < additional; ++i)
        sequence->blocks[sequence->block_count++] = cache->free_blocks[--cache->free_count];
    pthread_mutex_unlock(&cache->mutex);
    return 1;
}

int tensor_paged_kv_sequence_set_length(TensorPagedKVCache *cache,
                                        TensorPagedKVSequence handle,
                                        size_t token_length) {
    if (cache == NULL) return 0;
    pthread_mutex_lock(&cache->mutex);
    PagedSequence *sequence = find_sequence(cache, handle);
    size_t reserved = sequence == NULL ? 0 : sequence->block_count * cache->block_size;
    if (sequence == NULL || token_length > reserved) {
        pthread_mutex_unlock(&cache->mutex);
        return 0;
    }
    size_t needed = token_length / cache->block_size +
        (token_length % cache->block_size != 0);
    while (sequence->block_count > needed)
        cache->free_blocks[cache->free_count++] =
            sequence->blocks[--sequence->block_count];
    sequence->token_length = token_length;
    pthread_mutex_unlock(&cache->mutex);
    return 1;
}

int tensor_paged_kv_sequence_copy_block_table(TensorPagedKVCache *cache,
                                              TensorPagedKVSequence handle,
                                              uint32_t *block_ids,
                                              size_t block_id_capacity,
                                              size_t *block_count) {
    if (cache == NULL || block_count == NULL) return 0;
    pthread_mutex_lock(&cache->mutex);
    PagedSequence *sequence = find_sequence(cache, handle);
    if (sequence == NULL) {
        pthread_mutex_unlock(&cache->mutex);
        return 0;
    }
    *block_count = sequence->block_count;
    if (block_ids != NULL) {
        if (block_id_capacity < sequence->block_count) {
            pthread_mutex_unlock(&cache->mutex);
            return 0;
        }
        memcpy(block_ids, sequence->blocks,
               sequence->block_count * sizeof(*block_ids));
    }
    pthread_mutex_unlock(&cache->mutex);
    return 1;
}

int tensor_paged_kv_sequence_length(TensorPagedKVCache *cache,
                                    TensorPagedKVSequence handle,
                                    size_t *token_length) {
    if (cache == NULL || token_length == NULL) return 0;
    pthread_mutex_lock(&cache->mutex);
    PagedSequence *sequence = find_sequence(cache, handle);
    if (sequence == NULL) {
        pthread_mutex_unlock(&cache->mutex);
        return 0;
    }
    *token_length = sequence->token_length;
    pthread_mutex_unlock(&cache->mutex);
    return 1;
}

int tensor_paged_kv_sequence_release(TensorPagedKVCache *cache,
                                     TensorPagedKVSequence handle) {
    if (cache == NULL) return 0;
    pthread_mutex_lock(&cache->mutex);
    PagedSequence *sequence = find_sequence(cache, handle);
    if (sequence == NULL) {
        pthread_mutex_unlock(&cache->mutex);
        return 0;
    }
    if (sequence->decoding) {
        pthread_mutex_unlock(&cache->mutex);
        return 0;
    }
    void *backend_state = cache->backend_state;
    TensorPagedKVBackendSequenceRelease release_backend_sequence =
        cache->release_backend_sequence;
    for (size_t i = 0; i < sequence->block_count; ++i)
        cache->free_blocks[cache->free_count++] = sequence->blocks[i];
    sequence->block_count = 0;
    sequence->token_length = 0;
    sequence->active = 0;
    pthread_mutex_unlock(&cache->mutex);
    if (release_backend_sequence != NULL && backend_state != NULL)
        release_backend_sequence(backend_state, handle);
    return 1;
}

int tensor_paged_kv_sequence_begin_decode(TensorPagedKVCache *cache,
                                          TensorPagedKVSequence handle) {
    if (cache == NULL) return 0;
    pthread_mutex_lock(&cache->mutex);
    PagedSequence *sequence = find_sequence(cache, handle);
    if (sequence == NULL || sequence->decoding) {
        pthread_mutex_unlock(&cache->mutex);
        return 0;
    }
    sequence->decoding = 1;
    pthread_mutex_unlock(&cache->mutex);
    return 1;
}

void tensor_paged_kv_sequence_end_decode(TensorPagedKVCache *cache,
                                         TensorPagedKVSequence handle) {
    if (cache == NULL) return;
    pthread_mutex_lock(&cache->mutex);
    PagedSequence *sequence = find_sequence(cache, handle);
    if (sequence != NULL) sequence->decoding = 0;
    pthread_mutex_unlock(&cache->mutex);
}

size_t tensor_paged_kv_cache_free_blocks(TensorPagedKVCache *cache) {
    if (cache == NULL) return 0;
    pthread_mutex_lock(&cache->mutex);
    size_t count = cache->free_count;
    pthread_mutex_unlock(&cache->mutex);
    return count;
}

size_t tensor_paged_kv_cache_block_size(const TensorPagedKVCache *cache) {
    return cache == NULL ? 0 : cache->block_size;
}

size_t tensor_paged_kv_cache_block_count(const TensorPagedKVCache *cache) {
    return cache == NULL ? 0 : cache->block_count;
}

void *tensor_paged_kv_cache_backend_state(TensorPagedKVCache *cache,
                                          TensorPagedKVBackendStateKind kind) {
    if (cache == NULL) return NULL;
    pthread_mutex_lock(&cache->mutex);
    void *state = cache->backend_state_kind == kind ? cache->backend_state : NULL;
    pthread_mutex_unlock(&cache->mutex);
    return state;
}

int tensor_paged_kv_cache_set_backend_state(
    TensorPagedKVCache *cache, TensorPagedKVBackendStateKind kind, void *state,
    TensorPagedKVBackendStateDestroy destroy_state,
    TensorPagedKVBackendSequenceRelease release_sequence) {
    if (cache == NULL || state == NULL || destroy_state == NULL) return 0;
    pthread_mutex_lock(&cache->mutex);
    if (cache->backend_state != NULL) {
        pthread_mutex_unlock(&cache->mutex);
        return 0;
    }
    cache->backend_state_kind = kind;
    cache->backend_state = state;
    cache->destroy_backend_state = destroy_state;
    cache->release_backend_sequence = release_sequence;
    pthread_mutex_unlock(&cache->mutex);
    return 1;
}
