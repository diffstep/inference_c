#ifndef ALLOCATOR_H
#define ALLOCATOR_H

#include <stddef.h>

/*
 * Project-owned allocation wrappers. When USE_MIMALLOC is enabled
 * at build time, these use mimalloc; otherwise they use the C runtime.
 * Memory returned here must be released with allocator_free.
 */
void *allocator_malloc(size_t size);
void *allocator_calloc(size_t count, size_t size);
void *allocator_realloc(void *pointer, size_t size);
void allocator_free(void *pointer);
char *allocator_strdup(const char *string);

#endif
