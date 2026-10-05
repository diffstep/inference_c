#ifndef INFERENCE_SDK_METAL_LIBRARY_H
#define INFERENCE_SDK_METAL_LIBRARY_H

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <dispatch/dispatch.h>

#include <stdint.h>
#include <string.h>

#if defined(EMBED_METALLIBS)
extern const uint8_t *rust_attention_metallib_data(size_t *length);
extern const uint8_t *rust_tensor_metallib_data(size_t *length);
#endif

static inline id<MTLLibrary> metal_load_library(id<MTLDevice> device,
                                                const char *library_name,
                                                NSString *path,
                                                NSError **error) {
    id<MTLLibrary> library = nil;
#if defined(EMBED_METALLIBS)
    (void)path;
    size_t length = 0;
    const uint8_t *bytes = strcmp(library_name, "attention") == 0 ?
        rust_attention_metallib_data(&length) : rust_tensor_metallib_data(&length);
    dispatch_data_t data = bytes == NULL || length == 0 ? nil :
        dispatch_data_create(bytes, length, dispatch_get_main_queue(),
                             DISPATCH_DATA_DESTRUCTOR_DEFAULT);
    if (data != nil) library = [device newLibraryWithData:data error:error];
#else
    (void)library_name;
    library = [device newLibraryWithURL:[NSURL fileURLWithPath:path] error:error];
#endif
    return library;
}

#endif
