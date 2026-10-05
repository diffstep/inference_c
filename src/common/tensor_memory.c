#define _POSIX_C_SOURCE 200112L

#include "tensor_memory.h"

#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>

void *tensor_aligned_allocate(size_t bytes) {
    if (bytes == 0) return NULL;
    const long page_size_value = sysconf(_SC_PAGESIZE);
    if (page_size_value <= 0) return NULL;
    const size_t page_size = (size_t)page_size_value;
    if (bytes > SIZE_MAX - (page_size - 1)) return NULL;
    const size_t allocation_size = (bytes + page_size - 1) / page_size * page_size;
    void *allocation = NULL;
    if (posix_memalign(&allocation, page_size, allocation_size) != 0) return NULL;
    return allocation;
}
