#ifndef PAGED_KV_CACHE_H
#define PAGED_KV_CACHE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Metadata allocator for GPU-resident, fixed-size KV blocks. The block payload
 * belongs to the selected GPU backend; this object owns sequence block tables. */
typedef struct TensorPagedKVCache TensorPagedKVCache;
typedef uint64_t TensorPagedKVSequence;
typedef void (*TensorPagedKVBackendStateDestroy)(void *state);
typedef void (*TensorPagedKVBackendSequenceRelease)(void *state,
                                                     TensorPagedKVSequence sequence);
typedef enum {
    TENSOR_PAGED_KV_BACKEND_CUDA = 1,
    TENSOR_PAGED_KV_BACKEND_METAL = 2
} TensorPagedKVBackendStateKind;

int tensor_paged_kv_cache_create(size_t block_size, size_t block_count,
                                 size_t max_sequences,
                                 TensorPagedKVCache **cache);
void tensor_paged_kv_cache_free(TensorPagedKVCache *cache);
int tensor_paged_kv_sequence_create(TensorPagedKVCache *cache,
                                    TensorPagedKVSequence *sequence);
/* Reserves enough blocks for token_capacity without changing the visible length. */
int tensor_paged_kv_sequence_reserve(TensorPagedKVCache *cache,
                                     TensorPagedKVSequence sequence,
                                     size_t token_capacity);
/* Sets visible length and returns whole trailing blocks made unnecessary by
 * truncation. It must not exceed the currently reserved capacity. */
int tensor_paged_kv_sequence_set_length(TensorPagedKVCache *cache,
                                        TensorPagedKVSequence sequence,
                                        size_t token_length);
/* Copies a snapshot of the block table while holding the allocator lock.
 * Set block_ids to NULL to query the required block_count. */
int tensor_paged_kv_sequence_copy_block_table(TensorPagedKVCache *cache,
                                              TensorPagedKVSequence sequence,
                                              uint32_t *block_ids,
                                              size_t block_id_capacity,
                                              size_t *block_count);
int tensor_paged_kv_sequence_length(TensorPagedKVCache *cache,
                                    TensorPagedKVSequence sequence,
                                    size_t *token_length);
int tensor_paged_kv_sequence_release(TensorPagedKVCache *cache,
                                     TensorPagedKVSequence sequence);
/* Serializes decode transactions for a sequence. A sequence cannot be released
 * while a decode transaction is active. */
int tensor_paged_kv_sequence_begin_decode(TensorPagedKVCache *cache,
                                          TensorPagedKVSequence sequence);
void tensor_paged_kv_sequence_end_decode(TensorPagedKVCache *cache,
                                         TensorPagedKVSequence sequence);
size_t tensor_paged_kv_cache_free_blocks(TensorPagedKVCache *cache);
size_t tensor_paged_kv_cache_block_size(const TensorPagedKVCache *cache);
size_t tensor_paged_kv_cache_block_count(const TensorPagedKVCache *cache);
/* Internal backend attachment point; one selected GPU backend owns the state. */
void *tensor_paged_kv_cache_backend_state(TensorPagedKVCache *cache,
                                          TensorPagedKVBackendStateKind kind);
int tensor_paged_kv_cache_set_backend_state(
    TensorPagedKVCache *cache, TensorPagedKVBackendStateKind kind, void *state,
    TensorPagedKVBackendStateDestroy destroy_state,
    TensorPagedKVBackendSequenceRelease release_sequence);

#ifdef __cplusplus
}
#endif

#endif
