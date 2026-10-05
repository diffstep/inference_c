#include "paged_kv_cache.h"

#include <assert.h>
#include <stdint.h>
#include <stdio.h>

int main(void) {
    TensorPagedKVCache *cache = NULL;
    assert(tensor_paged_kv_cache_create(16, 5, 2, &cache));
    assert(tensor_paged_kv_cache_block_size(cache) == 16);
    assert(tensor_paged_kv_cache_free_blocks(cache) == 5);

    TensorPagedKVSequence a = 0, b = 0;
    assert(tensor_paged_kv_sequence_create(cache, &a));
    assert(tensor_paged_kv_sequence_create(cache, &b));
    assert(!tensor_paged_kv_sequence_create(cache, &(TensorPagedKVSequence){0}));

    /* A reservation crossing a page boundary gets exactly one more block. */
    assert(tensor_paged_kv_sequence_reserve(cache, a, 16));
    assert(tensor_paged_kv_sequence_set_length(cache, a, 16));
    assert(tensor_paged_kv_sequence_reserve(cache, a, 17));
    assert(tensor_paged_kv_cache_free_blocks(cache) == 3);
    uint32_t table[2] = {0};
    size_t table_count = 0;
    assert(tensor_paged_kv_sequence_copy_block_table(cache, a, NULL, 0,
                                                      &table_count));
    assert(table_count == 2);
    assert(!tensor_paged_kv_sequence_copy_block_table(cache, a, table, 1,
                                                       &table_count));
    assert(tensor_paged_kv_sequence_copy_block_table(cache, a, table, 2,
                                                      &table_count));
    assert(table_count == 2 && table[0] != table[1]);

    /* Failed reservation is atomic and does not alter either sequence. */
    assert(!tensor_paged_kv_sequence_reserve(cache, b, 49));
    assert(tensor_paged_kv_cache_free_blocks(cache) == 3);
    assert(tensor_paged_kv_sequence_reserve(cache, b, 32));
    assert(tensor_paged_kv_sequence_set_length(cache, b, 31));
    size_t length = 0;
    assert(tensor_paged_kv_sequence_length(cache, b, &length) && length == 31);
    assert(!tensor_paged_kv_sequence_set_length(cache, b, 33));
    assert(tensor_paged_kv_sequence_set_length(cache, b, 16));
    assert(tensor_paged_kv_cache_free_blocks(cache) == 2);
    assert(tensor_paged_kv_sequence_copy_block_table(cache, b, NULL, 0,
                                                      &table_count));
    assert(table_count == 1);
    assert(tensor_paged_kv_sequence_reserve(cache, b, 32));

    assert(tensor_paged_kv_sequence_begin_decode(cache, b));
    assert(!tensor_paged_kv_sequence_begin_decode(cache, b));
    assert(!tensor_paged_kv_sequence_release(cache, b));
    tensor_paged_kv_sequence_end_decode(cache, b);

    assert(tensor_paged_kv_sequence_release(cache, a));
    assert(!tensor_paged_kv_sequence_length(cache, a, &length));
    assert(tensor_paged_kv_cache_free_blocks(cache) == 3);
    TensorPagedKVSequence reused = 0;
    assert(tensor_paged_kv_sequence_create(cache, &reused));
    assert(reused != a); /* stale handles cannot name a newly created sequence */
    assert(!tensor_paged_kv_sequence_release(cache, a));
    assert(tensor_paged_kv_sequence_release(cache, reused));
    assert(tensor_paged_kv_sequence_release(cache, b));
    assert(tensor_paged_kv_cache_free_blocks(cache) == 5);

    tensor_paged_kv_cache_free(cache);
    puts("paged KV metadata allocator tests passed");
    return 0;
}
