#include "allocator.h"

#include <string.h>

#ifdef USE_MIMALLOC
#include <mimalloc.h>
#define BACKEND_MALLOC  mi_malloc
#define BACKEND_CALLOC  mi_calloc
#define BACKEND_REALLOC mi_realloc
#define BACKEND_FREE    mi_free
#else
#include <stdlib.h>
#define BACKEND_MALLOC  malloc
#define BACKEND_CALLOC  calloc
#define BACKEND_REALLOC realloc
#define BACKEND_FREE    free
#endif

void *allocator_malloc(size_t size) {
    return BACKEND_MALLOC(size);
}

void *allocator_calloc(size_t count, size_t size) {
    return BACKEND_CALLOC(count, size);
}

void *allocator_realloc(void *pointer, size_t size) {
    return BACKEND_REALLOC(pointer, size);
}

void allocator_free(void *pointer) {
    BACKEND_FREE(pointer);
}

char *allocator_strdup(const char *string) {
    if (string == NULL) return NULL;
    const size_t length = strlen(string);
    if (length == (size_t)-1) return NULL;
    char *copy = allocator_malloc(length + 1);
    if (copy != NULL) memcpy(copy, string, length + 1);
    return copy;
}
