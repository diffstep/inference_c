#include "safetensors.h"

#include <ctype.h>
#include <inttypes.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SMOLVLM_MAX_HEADER_SIZE (64u * 1024u * 1024u)
#define SMOLVLM_MAX_TENSOR_RANK 16u

typedef struct {
    SafeTensorInfo public_info;
    char *owned_name;
    uint64_t *owned_shape;
} StoredTensor;

struct SafeTensors {
    FILE *stream;
    char *path;
    uint64_t data_start;
    StoredTensor *tensors;
    size_t tensor_count;
};

typedef struct {
    const char *cursor;
    const char *end;
} JsonParser;

static void set_error(char *error, size_t capacity, const char *message) {
    if (error != NULL && capacity > 0) {
        snprintf(error, capacity, "%s", message);
    }
}

static char *copy_string(const char *value) {
    const size_t length = strlen(value);
    char *copy = malloc(length + 1);
    if (copy != NULL) {
        memcpy(copy, value, length + 1);
    }
    return copy;
}

static void skip_space(JsonParser *parser) {
    while (parser->cursor < parser->end && isspace((unsigned char)*parser->cursor)) {
        parser->cursor++;
    }
}

static int consume(JsonParser *parser, char expected) {
    skip_space(parser);
    if (parser->cursor >= parser->end || *parser->cursor != expected) {
        return 0;
    }
    parser->cursor++;
    return 1;
}

static int append_utf8(char *output, size_t capacity, size_t *length,
                       uint32_t codepoint) {
    unsigned char bytes[4];
    size_t count;
    if (codepoint <= 0x7f) {
        bytes[0] = (unsigned char)codepoint;
        count = 1;
    } else if (codepoint <= 0x7ff) {
        bytes[0] = (unsigned char)(0xc0 | (codepoint >> 6));
        bytes[1] = (unsigned char)(0x80 | (codepoint & 0x3f));
        count = 2;
    } else if (codepoint <= 0xffff) {
        bytes[0] = (unsigned char)(0xe0 | (codepoint >> 12));
        bytes[1] = (unsigned char)(0x80 | ((codepoint >> 6) & 0x3f));
        bytes[2] = (unsigned char)(0x80 | (codepoint & 0x3f));
        count = 3;
    } else if (codepoint <= 0x10ffff) {
        bytes[0] = (unsigned char)(0xf0 | (codepoint >> 18));
        bytes[1] = (unsigned char)(0x80 | ((codepoint >> 12) & 0x3f));
        bytes[2] = (unsigned char)(0x80 | ((codepoint >> 6) & 0x3f));
        bytes[3] = (unsigned char)(0x80 | (codepoint & 0x3f));
        count = 4;
    } else {
        return 0;
    }
    if (*length + count >= capacity) {
        return 0;
    }
    memcpy(output + *length, bytes, count);
    *length += count;
    return 1;
}

static int read_hex4(JsonParser *parser, uint32_t *value) {
    uint32_t result = 0;
    for (size_t index = 0; index < 4; index++) {
        if (parser->cursor >= parser->end) {
            return 0;
        }
        const unsigned char byte = (unsigned char)*parser->cursor++;
        result <<= 4;
        if (byte >= '0' && byte <= '9') {
            result |= (uint32_t)(byte - '0');
        } else if (byte >= 'a' && byte <= 'f') {
            result |= (uint32_t)(byte - 'a' + 10);
        } else if (byte >= 'A' && byte <= 'F') {
            result |= (uint32_t)(byte - 'A' + 10);
        } else {
            return 0;
        }
    }
    *value = result;
    return 1;
}

static char *read_string(JsonParser *parser) {
    skip_space(parser);
    if (parser->cursor >= parser->end || *parser->cursor++ != '"') {
        return NULL;
    }
    const size_t remaining = (size_t)(parser->end - parser->cursor);
    if (remaining == SIZE_MAX) {
        return NULL;
    }
    char *result = malloc(remaining + 1);
    if (result == NULL) {
        return NULL;
    }
    size_t length = 0;
    while (parser->cursor < parser->end) {
        unsigned char byte = (unsigned char)*parser->cursor++;
        if (byte == '"') {
            result[length] = '\0';
            return result;
        }
        if (byte < 0x20) {
            free(result);
            return NULL;
        }
        if (byte == '\\') {
            if (parser->cursor >= parser->end) {
                free(result);
                return NULL;
            }
            byte = (unsigned char)*parser->cursor++;
            switch (byte) {
                case '"':
                case '\\':
                case '/':
                    if (!append_utf8(result, remaining + 1, &length, byte)) {
                        free(result);
                        return NULL;
                    }
                    break;
                case 'b':
                    result[length++] = '\b';
                    break;
                case 'f':
                    result[length++] = '\f';
                    break;
                case 'n':
                    result[length++] = '\n';
                    break;
                case 'r':
                    result[length++] = '\r';
                    break;
                case 't':
                    result[length++] = '\t';
                    break;
                case 'u': {
                    uint32_t codepoint;
                    if (!read_hex4(parser, &codepoint)) {
                        free(result);
                        return NULL;
                    }
                    if (codepoint >= 0xd800 && codepoint <= 0xdbff) {
                        uint32_t low;
                        if (parser->end - parser->cursor < 6 ||
                            parser->cursor[0] != '\\' || parser->cursor[1] != 'u') {
                            free(result);
                            return NULL;
                        }
                        parser->cursor += 2;
                        if (!read_hex4(parser, &low) || low < 0xdc00 || low > 0xdfff) {
                            free(result);
                            return NULL;
                        }
                        codepoint = 0x10000 + ((codepoint - 0xd800) << 10) + (low - 0xdc00);
                    } else if (codepoint >= 0xdc00 && codepoint <= 0xdfff) {
                        free(result);
                        return NULL;
                    }
                    if (!append_utf8(result, remaining + 1, &length, codepoint)) {
                        free(result);
                        return NULL;
                    }
                    break;
                }
                default:
                    free(result);
                    return NULL;
            }
        } else {
            result[length++] = (char)byte;
        }
    }
    free(result);
    return NULL;
}

static int skip_value(JsonParser *parser, size_t depth) {
    if (depth > 64) {
        return 0;
    }
    skip_space(parser);
    if (parser->cursor >= parser->end) {
        return 0;
    }
    if (*parser->cursor == '"') {
        char *value = read_string(parser);
        if (value == NULL) {
            return 0;
        }
        free(value);
        return 1;
    }
    if (*parser->cursor == '{') {
        parser->cursor++;
        skip_space(parser);
        if (consume(parser, '}')) {
            return 1;
        }
        while (parser->cursor < parser->end) {
            char *key = read_string(parser);
            free(key);
            if (key == NULL || !consume(parser, ':') || !skip_value(parser, depth + 1)) {
                return 0;
            }
            if (consume(parser, '}')) {
                return 1;
            }
            if (!consume(parser, ',')) {
                return 0;
            }
        }
        return 0;
    }
    if (*parser->cursor == '[') {
        parser->cursor++;
        skip_space(parser);
        if (consume(parser, ']')) {
            return 1;
        }
        while (parser->cursor < parser->end) {
            if (!skip_value(parser, depth + 1)) {
                return 0;
            }
            if (consume(parser, ']')) {
                return 1;
            }
            if (!consume(parser, ',')) {
                return 0;
            }
        }
        return 0;
    }
    const char *start = parser->cursor;
    while (parser->cursor < parser->end &&
           !isspace((unsigned char)*parser->cursor) && *parser->cursor != ',' &&
           *parser->cursor != ']' && *parser->cursor != '}') {
        parser->cursor++;
    }
    return parser->cursor > start;
}

static int read_u64(JsonParser *parser, uint64_t *value) {
    skip_space(parser);
    if (parser->cursor >= parser->end || !isdigit((unsigned char)*parser->cursor)) {
        return 0;
    }
    uint64_t result = 0;
    while (parser->cursor < parser->end && isdigit((unsigned char)*parser->cursor)) {
        const uint64_t digit = (uint64_t)(*parser->cursor++ - '0');
        if (result > (UINT64_MAX - digit) / 10) {
            return 0;
        }
        result = result * 10 + digit;
    }
    *value = result;
    return 1;
}

static TensorDType parse_dtype(const char *name) {
    static const struct {
        const char *name;
        TensorDType dtype;
    } names[] = {
        {"BOOL", TENSOR_DTYPE_BOOL}, {"U8", TENSOR_DTYPE_U8},
        {"I8", TENSOR_DTYPE_I8}, {"I16", TENSOR_DTYPE_I16},
        {"U16", TENSOR_DTYPE_U16}, {"I32", TENSOR_DTYPE_I32},
        {"U32", TENSOR_DTYPE_U32}, {"I64", TENSOR_DTYPE_I64},
        {"U64", TENSOR_DTYPE_U64}, {"F8_E4M3", TENSOR_DTYPE_F8_E4M3},
        {"F8_E5M2", TENSOR_DTYPE_F8_E5M2}, {"F16", TENSOR_DTYPE_F16},
        {"BF16", TENSOR_DTYPE_BF16}, {"F32", TENSOR_DTYPE_F32},
        {"F64", TENSOR_DTYPE_F64},
    };
    for (size_t index = 0; index < sizeof(names) / sizeof(names[0]); index++) {
        if (strcmp(name, names[index].name) == 0) {
            return names[index].dtype;
        }
    }
    return TENSOR_DTYPE_UNKNOWN;
}

const char *tensor_dtype_name(TensorDType dtype) {
    switch (dtype) {
        case TENSOR_DTYPE_BOOL: return "BOOL";
        case TENSOR_DTYPE_U8: return "U8";
        case TENSOR_DTYPE_I8: return "I8";
        case TENSOR_DTYPE_I16: return "I16";
        case TENSOR_DTYPE_U16: return "U16";
        case TENSOR_DTYPE_I32: return "I32";
        case TENSOR_DTYPE_U32: return "U32";
        case TENSOR_DTYPE_I64: return "I64";
        case TENSOR_DTYPE_U64: return "U64";
        case TENSOR_DTYPE_F8_E4M3: return "F8_E4M3";
        case TENSOR_DTYPE_F8_E5M2: return "F8_E5M2";
        case TENSOR_DTYPE_F16: return "F16";
        case TENSOR_DTYPE_BF16: return "BF16";
        case TENSOR_DTYPE_F32: return "F32";
        case TENSOR_DTYPE_F64: return "F64";
        default: return "UNKNOWN";
    }
}

size_t tensor_dtype_size(TensorDType dtype) {
    switch (dtype) {
        case TENSOR_DTYPE_BOOL:
        case TENSOR_DTYPE_U8:
        case TENSOR_DTYPE_I8:
        case TENSOR_DTYPE_F8_E4M3:
        case TENSOR_DTYPE_F8_E5M2:
            return 1;
        case TENSOR_DTYPE_I16:
        case TENSOR_DTYPE_U16:
        case TENSOR_DTYPE_F16:
        case TENSOR_DTYPE_BF16:
            return 2;
        case TENSOR_DTYPE_I32:
        case TENSOR_DTYPE_U32:
        case TENSOR_DTYPE_F32:
            return 4;
        case TENSOR_DTYPE_I64:
        case TENSOR_DTYPE_U64:
        case TENSOR_DTYPE_F64:
            return 8;
        default:
            return 0;
    }
}

static int parse_shape(JsonParser *parser, uint64_t **shape, size_t *rank) {
    if (!consume(parser, '[')) {
        return 0;
    }
    uint64_t *values = NULL;
    size_t count = 0;
    skip_space(parser);
    if (!consume(parser, ']')) {
        while (count < SMOLVLM_MAX_TENSOR_RANK) {
            uint64_t dimension;
            if (!read_u64(parser, &dimension)) {
                free(values);
                return 0;
            }
            uint64_t *next = realloc(values, (count + 1) * sizeof(*values));
            if (next == NULL) {
                free(values);
                return 0;
            }
            values = next;
            values[count++] = dimension;
            if (consume(parser, ']')) {
                *shape = values;
                *rank = count;
                return 1;
            }
            if (!consume(parser, ',')) {
                free(values);
                return 0;
            }
        }
        free(values);
        return 0;
    }
    *shape = NULL;
    *rank = 0;
    return 1;
}

static int parse_offsets(JsonParser *parser, uint64_t offsets[2]) {
    if (!consume(parser, '[') || !read_u64(parser, &offsets[0]) ||
        !consume(parser, ',') || !read_u64(parser, &offsets[1]) ||
        !consume(parser, ']')) {
        return 0;
    }
    return 1;
}

static int parse_tensor(JsonParser *parser, const char *name, StoredTensor *tensor) {
    if (!consume(parser, '{')) {
        return 0;
    }
    char *dtype_name = NULL;
    uint64_t *shape = NULL;
    size_t rank = 0;
    uint64_t offsets[2] = {0, 0};
    int got_dtype = 0;
    int got_shape = 0;
    int got_offsets = 0;
    skip_space(parser);
    if (!consume(parser, '}')) {
        while (parser->cursor < parser->end) {
            char *key = read_string(parser);
            if (key == NULL || !consume(parser, ':')) {
                free(key);
                goto failure;
            }
            int ok = 0;
            if (strcmp(key, "dtype") == 0) {
                dtype_name = read_string(parser);
                ok = dtype_name != NULL;
                got_dtype = ok;
            } else if (strcmp(key, "shape") == 0) {
                ok = parse_shape(parser, &shape, &rank);
                got_shape = ok;
            } else if (strcmp(key, "data_offsets") == 0) {
                ok = parse_offsets(parser, offsets);
                got_offsets = ok;
            } else {
                ok = skip_value(parser, 0);
            }
            free(key);
            if (!ok) {
                goto failure;
            }
            if (consume(parser, '}')) {
                break;
            }
            if (!consume(parser, ',')) {
                goto failure;
            }
        }
    }
    if (!got_dtype || !got_shape || !got_offsets || offsets[1] < offsets[0]) {
        goto failure;
    }
    const TensorDType dtype = parse_dtype(dtype_name);
    const size_t element_size = tensor_dtype_size(dtype);
    if (element_size == 0) {
        goto failure;
    }
    uint64_t elements = 1;
    for (size_t index = 0; index < rank; index++) {
        if (shape[index] != 0 && elements > UINT64_MAX / shape[index]) {
            goto failure;
        }
        elements *= shape[index];
    }
    if (elements > UINT64_MAX / element_size ||
        elements * element_size != offsets[1] - offsets[0]) {
        goto failure;
    }
    tensor->owned_name = copy_string(name);
    if (tensor->owned_name == NULL) {
        goto failure;
    }
    tensor->owned_shape = shape;
    tensor->public_info = (SafeTensorInfo){
        .name = tensor->owned_name,
        .dtype = dtype,
        .rank = rank,
        .shape = tensor->owned_shape,
        .data_offset = offsets[0],
        .byte_length = offsets[1] - offsets[0],
    };
    free(dtype_name);
    return 1;

failure:
    free(dtype_name);
    free(shape);
    return 0;
}

static int append_tensor(SafeTensors *file, StoredTensor *tensor) {
    if (file->tensor_count == SIZE_MAX / sizeof(*file->tensors)) {
        return 0;
    }
    StoredTensor *next = realloc(file->tensors,
                                 (file->tensor_count + 1) * sizeof(*file->tensors));
    if (next == NULL) {
        return 0;
    }
    file->tensors = next;
    file->tensors[file->tensor_count++] = *tensor;
    return 1;
}

static int parse_header(SafeTensors *file, char *header, size_t length) {
    JsonParser parser = {header, header + length};
    if (!consume(&parser, '{')) {
        return 0;
    }
    skip_space(&parser);
    if (consume(&parser, '}')) {
        return parser.cursor == parser.end;
    }
    while (parser.cursor < parser.end) {
        char *name = read_string(&parser);
        if (name == NULL || !consume(&parser, ':')) {
            free(name);
            return 0;
        }
        int ok;
        if (strcmp(name, "__metadata__") == 0) {
            ok = skip_value(&parser, 0);
        } else {
            StoredTensor tensor = {0};
            ok = parse_tensor(&parser, name, &tensor) && append_tensor(file, &tensor);
            if (!ok) {
                free(tensor.owned_name);
                free(tensor.owned_shape);
            }
        }
        free(name);
        if (!ok) {
            return 0;
        }
        if (consume(&parser, '}')) {
            skip_space(&parser);
            return parser.cursor == parser.end;
        }
        if (!consume(&parser, ',')) {
            return 0;
        }
    }
    return 0;
}

int safetensors_open(const char *path, SafeTensors **result,
                     char *error, size_t error_capacity) {
    if (path == NULL || result == NULL) {
        set_error(error, error_capacity, "invalid argument");
        return 0;
    }
    *result = NULL;
    FILE *stream = fopen(path, "rb");
    if (stream == NULL) {
        set_error(error, error_capacity, "cannot open safetensors file");
        return 0;
    }
    if (fseek(stream, 0, SEEK_END) != 0) {
        fclose(stream);
        set_error(error, error_capacity, "cannot seek safetensors file");
        return 0;
    }
    const long file_length = ftell(stream);
    if (file_length < 8 || fseek(stream, 0, SEEK_SET) != 0) {
        fclose(stream);
        set_error(error, error_capacity, "safetensors file is too short or not seekable");
        return 0;
    }
    unsigned char length_bytes[8];
    if (fread(length_bytes, 1, sizeof(length_bytes), stream) != sizeof(length_bytes)) {
        fclose(stream);
        set_error(error, error_capacity, "cannot read safetensors header length");
        return 0;
    }
    uint64_t header_length = 0;
    for (size_t index = 0; index < sizeof(length_bytes); index++) {
        header_length |= (uint64_t)length_bytes[index] << (index * 8);
    }
    if (header_length == 0 || header_length > SMOLVLM_MAX_HEADER_SIZE ||
        header_length > (uint64_t)file_length - 8) {
        fclose(stream);
        set_error(error, error_capacity, "invalid safetensors header length");
        return 0;
    }
    char *header = malloc((size_t)header_length + 1);
    SafeTensors *file = calloc(1, sizeof(*file));
    if (header == NULL || file == NULL) {
        free(header);
        free(file);
        fclose(stream);
        set_error(error, error_capacity, "out of memory reading safetensors header");
        return 0;
    }
    if (fread(header, 1, (size_t)header_length, stream) != header_length) {
        free(header);
        free(file);
        fclose(stream);
        set_error(error, error_capacity, "cannot read safetensors header");
        return 0;
    }
    header[header_length] = '\0';
    file->stream = stream;
    file->data_start = 8 + header_length;
    file->path = copy_string(path);
    const int parsed = file->path != NULL &&
                       parse_header(file, header, (size_t)header_length);
    free(header);
    if (!parsed) {
        safetensors_close(file);
        set_error(error, error_capacity, "invalid or unsupported safetensors header");
        return 0;
    }
    for (size_t index = 0; index < file->tensor_count; index++) {
        const SafeTensorInfo *tensor = &file->tensors[index].public_info;
        if (tensor->data_offset > (uint64_t)file_length - file->data_start ||
            tensor->byte_length > (uint64_t)file_length - file->data_start - tensor->data_offset) {
            safetensors_close(file);
            set_error(error, error_capacity, "tensor data range exceeds file length");
            return 0;
        }
    }
    *result = file;
    return 1;
}

void safetensors_close(SafeTensors *file) {
    if (file == NULL) {
        return;
    }
    if (file->stream != NULL) {
        fclose(file->stream);
    }
    for (size_t index = 0; index < file->tensor_count; index++) {
        free(file->tensors[index].owned_name);
        free(file->tensors[index].owned_shape);
    }
    free(file->tensors);
    free(file->path);
    free(file);
}

size_t safetensors_count(const SafeTensors *file) {
    return file == NULL ? 0 : file->tensor_count;
}

const SafeTensorInfo *safetensors_at(const SafeTensors *file, size_t index) {
    if (file == NULL || index >= file->tensor_count) {
        return NULL;
    }
    return &file->tensors[index].public_info;
}

const SafeTensorInfo *safetensors_find(const SafeTensors *file, const char *name) {
    if (file == NULL || name == NULL) {
        return NULL;
    }
    for (size_t index = 0; index < file->tensor_count; index++) {
        const SafeTensorInfo *tensor = &file->tensors[index].public_info;
        if (strcmp(tensor->name, name) == 0) {
            return tensor;
        }
    }
    return NULL;
}

int safetensors_read_raw(const SafeTensors *file, const SafeTensorInfo *tensor,
                         void *destination, size_t capacity,
                         char *error, size_t error_capacity) {
    if (tensor == NULL || tensor->byte_length > SIZE_MAX) {
        set_error(error, error_capacity, "invalid tensor read arguments or destination size");
        return 0;
    }
    if (tensor->byte_length > capacity) {
        set_error(error, error_capacity, "invalid tensor read arguments or destination size");
        return 0;
    }
    return safetensors_read_raw_range(file, tensor, 0, destination,
        (size_t)tensor->byte_length, error, error_capacity);
}

int safetensors_read_raw_range(const SafeTensors *file,
                               const SafeTensorInfo *tensor,
                               uint64_t byte_offset, void *destination,
                               size_t byte_count, char *error,
                               size_t error_capacity) {
    int belongs_to_file = 0;
    if (file != NULL && tensor != NULL) {
        for (size_t index = 0; index < file->tensor_count; index++) {
            if (tensor == &file->tensors[index].public_info) {
                belongs_to_file = 1;
                break;
            }
        }
    }
    if (!belongs_to_file || destination == NULL ||
        byte_offset > tensor->byte_length ||
        byte_count > tensor->byte_length - byte_offset ||
        file->data_start > LONG_MAX ||
        tensor->data_offset > (uint64_t)LONG_MAX - file->data_start ||
        byte_offset > (uint64_t)LONG_MAX - file->data_start - tensor->data_offset) {
        set_error(error, error_capacity, "invalid tensor read arguments or destination size");
        return 0;
    }
    const long position = (long)(file->data_start + tensor->data_offset + byte_offset);
    if (fseek(file->stream, position, SEEK_SET) != 0 ||
        fread(destination, 1, byte_count, file->stream) != byte_count) {
        set_error(error, error_capacity, "failed to read tensor data");
        return 0;
    }
    return 1;
}
