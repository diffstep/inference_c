#ifndef TENSOR_WORKSPACE_H
#define TENSOR_WORKSPACE_H

#include <stddef.h>

typedef struct TensorWorkspace TensorWorkspace;

typedef struct {
    size_t block_index;
    size_t used_bytes;
} TensorWorkspaceMark;

int tensor_workspace_create(size_t initial_block_size,
                            TensorWorkspace **result, char *error,
                            size_t error_capacity);
void tensor_workspace_free(TensorWorkspace *workspace);
void *tensor_workspace_allocate(TensorWorkspace *workspace, size_t size,
                                size_t alignment, char *error,
                                size_t error_capacity);
TensorWorkspaceMark tensor_workspace_mark(const TensorWorkspace *workspace);
int tensor_workspace_release(TensorWorkspace *workspace,
                             TensorWorkspaceMark mark, char *error,
                             size_t error_capacity);
void tensor_workspace_reset(TensorWorkspace *workspace);
size_t tensor_workspace_used_bytes(const TensorWorkspace *workspace);
size_t tensor_workspace_reserved_bytes(const TensorWorkspace *workspace);
size_t tensor_workspace_peak_bytes(const TensorWorkspace *workspace);

#endif
