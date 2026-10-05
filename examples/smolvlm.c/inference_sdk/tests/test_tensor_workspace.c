#include "tensor_workspace.h"
#include "tensor_buffer.h"

#include <assert.h>
#include <stdint.h>
#include <stdio.h>

int main(void) {
    TensorWorkspace *workspace = NULL;
    char error[256] = {0};
    assert(tensor_workspace_create(64, &workspace, error, sizeof(error)));
    void *first = tensor_workspace_allocate(workspace, 13, 8, error, sizeof(error));
    assert(first != NULL && ((uintptr_t)first % 8) == 0);
    const TensorWorkspaceMark mark = tensor_workspace_mark(workspace);
    void *large = tensor_workspace_allocate(workspace, 128, 32, error, sizeof(error));
    assert(large != NULL && ((uintptr_t)large % 32) == 0);
    assert(tensor_workspace_reserved_bytes(workspace) >= 192);
    assert(tensor_workspace_used_bytes(workspace) > 13);
    const size_t peak = tensor_workspace_peak_bytes(workspace);
    assert(peak == tensor_workspace_used_bytes(workspace));
    assert(tensor_workspace_release(workspace, mark, error, sizeof(error)));
    assert(tensor_workspace_used_bytes(workspace) == mark.used_bytes);
    void *reused = tensor_workspace_allocate(workspace, 128, 32, error, sizeof(error));
    assert(reused == large);
    assert(tensor_workspace_peak_bytes(workspace) == peak);
    tensor_workspace_reset(workspace);
    assert(tensor_workspace_used_bytes(workspace) == 0);
    assert(tensor_workspace_reserved_bytes(workspace) >= 192);
    void *after_reset = tensor_workspace_allocate(workspace, 13, 8, error,
                                                  sizeof(error));
    assert(after_reset == first);
    assert(tensor_workspace_allocate(workspace, 3, 3, error, sizeof(error)) == NULL);
    assert(!tensor_workspace_release(workspace, (TensorWorkspaceMark){99, 0},
                                     error, sizeof(error)));

    tensor_workspace_reset(workspace);
    const uint64_t tensor_shape[] = {16};
    TensorBuffer *workspace_tensor = NULL;
    assert(tensor_buffer_create_in_workspace(TENSOR_DTYPE_F32, tensor_shape, 1,
                                             workspace, &workspace_tensor,
                                             error, sizeof(error)));
    void *tensor_payload = tensor_buffer_data_mut(workspace_tensor);
    assert(tensor_payload != NULL);
    tensor_buffer_free(workspace_tensor);
    tensor_workspace_reset(workspace);
    assert(tensor_buffer_create_in_workspace(TENSOR_DTYPE_F32, tensor_shape, 1,
                                             workspace, &workspace_tensor,
                                             error, sizeof(error)));
    assert(tensor_buffer_data(workspace_tensor) == tensor_payload);
    tensor_buffer_free(workspace_tensor);
    tensor_workspace_free(workspace);
    puts("tensor workspace tests passed");
    return 0;
}
