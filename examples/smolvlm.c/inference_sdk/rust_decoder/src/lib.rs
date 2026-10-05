use image::ImageReader;
use std::ffi::{c_char, c_void, CStr};
use std::path::Path;
use tokenizers::Tokenizer;

type ImageAllocate = unsafe extern "C" fn(usize) -> *mut u8;
type TokenizerAllocate = unsafe extern "C" fn(usize) -> *mut c_void;

unsafe fn write_error(error: *mut c_char, capacity: usize, message: &str) {
    if error.is_null() || capacity == 0 { return; }
    let bytes = message.as_bytes();
    let length = bytes.len().min(capacity - 1);
    std::ptr::copy_nonoverlapping(bytes.as_ptr(), error.cast(), length);
    *error.add(length) = 0;
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rust_tokenizer_open(path: *const c_char, result: *mut *mut c_void, error: *mut c_char, error_capacity: usize) -> i32 {
    if !result.is_null() { *result = std::ptr::null_mut(); }
    if path.is_null() || result.is_null() {
        write_error(error, error_capacity, "invalid tokenizer open arguments");
        return 0;
    }
    let Ok(path) = CStr::from_ptr(path).to_str() else {
        write_error(error, error_capacity, "tokenizer path is not valid UTF-8");
        return 0;
    };
    match Tokenizer::from_file(path) {
        Ok(tokenizer) => { *result = Box::into_raw(Box::new(tokenizer)).cast(); 1 }
        Err(message) => { write_error(error, error_capacity, &message.to_string()); 0 }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rust_tokenizer_close(tokenizer: *mut c_void) {
    if !tokenizer.is_null() { drop(Box::from_raw(tokenizer.cast::<Tokenizer>())); }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rust_tokenizer_encode(tokenizer: *const c_void, text: *const c_char, allocate: TokenizerAllocate, ids: *mut *mut u32, count: *mut usize, error: *mut c_char, error_capacity: usize) -> i32 {
    if !ids.is_null() { *ids = std::ptr::null_mut(); }
    if !count.is_null() { *count = 0; }
    if tokenizer.is_null() || text.is_null() || ids.is_null() || count.is_null() {
        write_error(error, error_capacity, "invalid tokenizer encode arguments");
        return 0;
    }
    let Ok(text) = CStr::from_ptr(text).to_str() else {
        write_error(error, error_capacity, "input text is not valid UTF-8");
        return 0;
    };
    let tokenizer = &*tokenizer.cast::<Tokenizer>();
    let encoding = match tokenizer.encode(text, false) {
        Ok(encoding) => encoding,
        Err(message) => { write_error(error, error_capacity, &message.to_string()); return 0; }
    };
    let token_ids = encoding.get_ids();
    if token_ids.is_empty() { return 1; }
    let Some(size) = token_ids.len().checked_mul(std::mem::size_of::<u32>()) else {
        write_error(error, error_capacity, "encoded token buffer size overflow");
        return 0;
    };
    let output = allocate(size).cast::<u32>();
    if output.is_null() { write_error(error, error_capacity, "out of memory encoding text"); return 0; }
    std::ptr::copy_nonoverlapping(token_ids.as_ptr(), output, token_ids.len());
    *ids = output;
    *count = token_ids.len();
    1
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rust_tokenizer_decode(tokenizer: *const c_void, ids: *const u32, count: usize, allocate: TokenizerAllocate, text: *mut *mut c_char, error: *mut c_char, error_capacity: usize) -> i32 {
    if !text.is_null() { *text = std::ptr::null_mut(); }
    if tokenizer.is_null() || (count != 0 && ids.is_null()) || text.is_null() {
        write_error(error, error_capacity, "invalid tokenizer decode arguments");
        return 0;
    }
    let tokenizer = &*tokenizer.cast::<Tokenizer>();
    let token_ids = if count == 0 { &[] } else { std::slice::from_raw_parts(ids, count) };
    let decoded = match tokenizer.decode(token_ids, true) {
        Ok(decoded) => decoded,
        Err(message) => { write_error(error, error_capacity, &message.to_string()); return 0; }
    };
    let Some(size) = decoded.len().checked_add(1) else {
        write_error(error, error_capacity, "decoded text size overflow");
        return 0;
    };
    let output = allocate(size).cast::<u8>();
    if output.is_null() { write_error(error, error_capacity, "out of memory decoding tokens"); return 0; }
    std::ptr::copy_nonoverlapping(decoded.as_ptr(), output, decoded.len());
    *output.add(decoded.len()) = 0;
    *text = output.cast();
    1
}

#[cfg(feature = "embedded-metallibs")]
static ATTENTION_METALLIB: &[u8] = include_bytes!(concat!(env!("INFERENCE_SDK_METALLIB_DIR"), "/attention.metallib"));
#[cfg(feature = "embedded-metallibs")]
static TENSOR_METALLIB: &[u8] = include_bytes!(concat!(env!("INFERENCE_SDK_METALLIB_DIR"), "/tensor_kernels.metallib"));

#[cfg(feature = "embedded-metallibs")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rust_attention_metallib_data(length: *mut usize) -> *const u8 {
    if length.is_null() {
        return std::ptr::null();
    }
    *length = ATTENTION_METALLIB.len();
    ATTENTION_METALLIB.as_ptr()
}

#[cfg(feature = "embedded-metallibs")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rust_tensor_metallib_data(length: *mut usize) -> *const u8 {
    if length.is_null() {
        return std::ptr::null();
    }
    *length = TENSOR_METALLIB.len();
    TENSOR_METALLIB.as_ptr()
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn image_decode_rgb_rust(
    path: *const c_char,
    allocate: ImageAllocate,
    pixels: *mut *mut u8,
    width: *mut usize,
    height: *mut usize,
) -> i32 {
    if path.is_null() || pixels.is_null() || width.is_null() || height.is_null() {
        return 0;
    }
    *pixels = std::ptr::null_mut();
    let Ok(path) = CStr::from_ptr(path).to_str() else {
        return 0;
    };
    let Ok(image) = ImageReader::open(Path::new(path))
        .and_then(|reader| reader.decode().map_err(std::io::Error::other))
    else {
        return 0;
    };
    let rgb = image.to_rgb8();
    let size = rgb.as_raw().len();
    let output = allocate(size) as *mut c_void as *mut u8;
    if output.is_null() {
        return 0;
    }
    std::ptr::copy_nonoverlapping(rgb.as_raw().as_ptr(), output, size);
    *width = rgb.width() as usize;
    *height = rgb.height() as usize;
    *pixels = output;
    1
}
