#include "tensor_workspace.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct {
    unsigned char *data;
    size_t capacity;
    size_t used;
} WorkspaceBlock;

struct TensorWorkspace {
    WorkspaceBlock *blocks;
    size_t block_count;
    size_t block_capacity;
    size_t active_block;
    size_t initial_block_size;
    size_t reserved_bytes;
    size_t peak_bytes;
};

static void set_error(char *error, size_t capacity, const char *message) {
    if (error != NULL && capacity > 0) snprintf(error, capacity, "%s", message);
}

static int is_power_of_two(size_t value) {
    return value != 0 && (value & (value - 1)) == 0;
}

static size_t used_bytes(const TensorWorkspace *workspace) {
    size_t total = 0;
    for (size_t index = 0; index < workspace->block_count; index++)
        total += workspace->blocks[index].used;
    return total;
}

static int add_block(TensorWorkspace *workspace, size_t minimum_capacity,
                     char *error, size_t error_capacity) {
    size_t capacity = workspace->initial_block_size;
    if (workspace->block_count > 0) {
        const size_t previous = workspace->blocks[workspace->block_count - 1].capacity;
        if (previous <= SIZE_MAX / 2 && previous * 2 > capacity)
            capacity = previous * 2;
    }
    if (capacity < minimum_capacity) capacity = minimum_capacity;
    if (capacity > SIZE_MAX - workspace->reserved_bytes) {
        set_error(error, error_capacity, "workspace capacity exceeds addressable size");
        return 0;
    }
    if (workspace->block_count == workspace->block_capacity) {
        size_t next_capacity = workspace->block_capacity == 0 ? 4 :
                               workspace->block_capacity * 2;
        if (next_capacity < workspace->block_capacity ||
            next_capacity > SIZE_MAX / sizeof(*workspace->blocks)) {
            set_error(error, error_capacity, "too many workspace blocks");
            return 0;
        }
        WorkspaceBlock *next = realloc(workspace->blocks,
                                       next_capacity * sizeof(*next));
        if (next == NULL) {
            set_error(error, error_capacity, "out of memory growing workspace block list");
            return 0;
        }
        workspace->blocks = next;
        workspace->block_capacity = next_capacity;
    }
    unsigned char *data = malloc(capacity);
    if (data == NULL) {
        set_error(error, error_capacity, "out of memory allocating workspace block");
        return 0;
    }
    workspace->blocks[workspace->block_count++] =
        (WorkspaceBlock){data, capacity, 0};
    workspace->reserved_bytes += capacity;
    return 1;
}

int tensor_workspace_create(size_t initial_block_size,
                            TensorWorkspace **result, char *error,
                            size_t error_capacity) {
    if (result != NULL) *result = NULL;
    if (result == NULL || initial_block_size == 0) {
        set_error(error, error_capacity, "workspace requires a result and nonzero block size");
        return 0;
    }
    TensorWorkspace *workspace = calloc(1, sizeof(*workspace));
    if (workspace == NULL) {
        set_error(error, error_capacity, "out of memory creating workspace");
        return 0;
    }
    workspace->initial_block_size = initial_block_size;
    *result = workspace;
    return 1;
}

void tensor_workspace_free(TensorWorkspace *workspace) {
    if (workspace == NULL) return;
    for (size_t index = 0; index < workspace->block_count; index++)
        free(workspace->blocks[index].data);
    free(workspace->blocks);
    free(workspace);
}

void *tensor_workspace_allocate(TensorWorkspace *workspace, size_t size,
                                size_t alignment, char *error,
                                size_t error_capacity) {
    if (workspace == NULL || !is_power_of_two(alignment)) {
        set_error(error, error_capacity, "workspace allocation requires power-of-two alignment");
        return NULL;
    }
    if (size == 0) size = 1;
    for (size_t index = workspace->active_block; index < workspace->block_count; index++) {
        WorkspaceBlock *block = &workspace->blocks[index];
        const uintptr_t address = (uintptr_t)(block->data + block->used);
        const size_t padding = (size_t)((-(uintptr_t)address) & (alignment - 1));
        if (padding <= block->capacity - block->used &&
            size <= block->capacity - block->used - padding) {
            block->used += padding;
            void *result = block->data + block->used;
            block->used += size;
            workspace->active_block = index;
            const size_t current = used_bytes(workspace);
            if (current > workspace->peak_bytes) workspace->peak_bytes = current;
            return result;
        }
    }
    if (size > SIZE_MAX - (alignment - 1)) {
        set_error(error, error_capacity, "workspace allocation size overflow");
        return NULL;
    }
    if (!add_block(workspace, size + alignment - 1, error, error_capacity))
        return NULL;
    workspace->active_block = workspace->block_count - 1;
    return tensor_workspace_allocate(workspace, size, alignment, error,
                                     error_capacity);
}

TensorWorkspaceMark tensor_workspace_mark(const TensorWorkspace *workspace) {
    TensorWorkspaceMark mark = {0, 0};
    if (workspace == NULL) return mark;
    mark.block_index = workspace->active_block;
    if (workspace->block_count > 0)
        mark.used_bytes = workspace->blocks[workspace->active_block].used;
    return mark;
}

int tensor_workspace_release(TensorWorkspace *workspace,
                             TensorWorkspaceMark mark, char *error,
                             size_t error_capacity) {
    if (workspace == NULL || mark.block_index >= workspace->block_count ||
        mark.used_bytes > workspace->blocks[mark.block_index].used) {
        set_error(error, error_capacity, "workspace mark is invalid");
        return 0;
    }
    workspace->blocks[mark.block_index].used = mark.used_bytes;
    for (size_t index = mark.block_index + 1; index < workspace->block_count; index++)
        workspace->blocks[index].used = 0;
    workspace->active_block = mark.block_index;
    return 1;
}

void tensor_workspace_reset(TensorWorkspace *workspace) {
    if (workspace == NULL) return;
    for (size_t index = 0; index < workspace->block_count; index++)
        workspace->blocks[index].used = 0;
    workspace->active_block = 0;
}

size_t tensor_workspace_used_bytes(const TensorWorkspace *workspace) {
    return workspace == NULL ? 0 : used_bytes(workspace);
}

size_t tensor_workspace_reserved_bytes(const TensorWorkspace *workspace) {
    return workspace == NULL ? 0 : workspace->reserved_bytes;
}

size_t tensor_workspace_peak_bytes(const TensorWorkspace *workspace) {
    return workspace == NULL ? 0 : workspace->peak_bytes;
}
