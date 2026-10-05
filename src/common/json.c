#include "json.h"

#include <ctype.h>
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define JSON_MAX_DEPTH 128u
#define JSON_MAX_FILE_SIZE (64u * 1024u * 1024u)

typedef struct {
    JsonValue **items;
    size_t count;
    size_t capacity;
} JsonArray;

typedef struct {
    char **keys;
    JsonValue **values;
    size_t count;
    size_t capacity;
    size_t *slots;
    size_t slot_count;
} JsonObject;

struct JsonValue {
    JsonType type;
    union {
        int boolean;
        struct {
            double floating;
            char *text;
        } number;
        char *string;
        JsonArray array;
        JsonObject object;
    } data;
};

typedef struct {
    const char *cursor;
    const char *end;
    char *error;
    size_t error_capacity;
} JsonParser;

static void set_error(JsonParser *parser, const char *message) {
    if (parser->error != NULL && parser->error_capacity > 0 && parser->error[0] == '\0') {
        snprintf(parser->error, parser->error_capacity, "%s", message);
    }
}

static void skip_space(JsonParser *parser) {
    while (parser->cursor < parser->end && isspace((unsigned char)*parser->cursor)) {
        parser->cursor++;
    }
}

static JsonValue *new_value(JsonType type) {
    JsonValue *value = calloc(1, sizeof(*value));
    if (value != NULL) {
        value->type = type;
    }
    return value;
}

static int append_codepoint(char *output, size_t capacity, size_t *length,
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

static int read_hex4(JsonParser *parser, uint32_t *result) {
    uint32_t value = 0;
    for (size_t index = 0; index < 4; index++) {
        if (parser->cursor >= parser->end) {
            return 0;
        }
        const unsigned char digit = (unsigned char)*parser->cursor++;
        value <<= 4;
        if (digit >= '0' && digit <= '9') {
            value |= (uint32_t)(digit - '0');
        } else if (digit >= 'a' && digit <= 'f') {
            value |= (uint32_t)(digit - 'a' + 10);
        } else if (digit >= 'A' && digit <= 'F') {
            value |= (uint32_t)(digit - 'A' + 10);
        } else {
            return 0;
        }
    }
    *result = value;
    return 1;
}

static char *parse_string(JsonParser *parser) {
    skip_space(parser);
    if (parser->cursor >= parser->end || *parser->cursor++ != '"') {
        return NULL;
    }
    const char *string_end = parser->cursor;
    while (string_end < parser->end && *string_end != '"') {
        if (*string_end == '\\' && string_end + 1 < parser->end) string_end++;
        string_end++;
    }
    const size_t capacity = (size_t)(string_end - parser->cursor) + 1;
    char *result = malloc(capacity);
    if (result == NULL) {
        set_error(parser, "out of memory parsing JSON string");
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
        if (byte != '\\') {
            result[length++] = (char)byte;
            continue;
        }
        if (parser->cursor >= parser->end) {
            free(result);
            return NULL;
        }
        byte = (unsigned char)*parser->cursor++;
        switch (byte) {
            case '"':
            case '\\':
            case '/':
                result[length++] = (char)byte;
                break;
            case 'b': result[length++] = '\b'; break;
            case 'f': result[length++] = '\f'; break;
            case 'n': result[length++] = '\n'; break;
            case 'r': result[length++] = '\r'; break;
            case 't': result[length++] = '\t'; break;
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
                if (!append_codepoint(result, capacity, &length, codepoint)) {
                    free(result);
                    return NULL;
                }
                break;
            }
            default:
                free(result);
                return NULL;
        }
    }
    free(result);
    return NULL;
}

void json_free(JsonValue *value) {
    if (value == NULL) {
        return;
    }
    if (value->type == JSON_STRING) {
        free(value->data.string);
    } else if (value->type == JSON_NUMBER) {
        free(value->data.number.text);
    } else if (value->type == JSON_ARRAY) {
        for (size_t index = 0; index < value->data.array.count; index++) {
            json_free(value->data.array.items[index]);
        }
        free(value->data.array.items);
    } else if (value->type == JSON_OBJECT) {
        for (size_t index = 0; index < value->data.object.count; index++) {
            free(value->data.object.keys[index]);
            json_free(value->data.object.values[index]);
        }
        free(value->data.object.keys);
        free(value->data.object.values);
        free(value->data.object.slots);
    }
    free(value);
}

static int append_array(JsonValue *array, JsonValue *item) {
    if (array->data.array.count == array->data.array.capacity) {
        size_t capacity = array->data.array.capacity == 0 ? 8 : array->data.array.capacity * 2;
        if (capacity < array->data.array.capacity || capacity > SIZE_MAX / sizeof(*array->data.array.items))
            return 0;
        JsonValue **items = realloc(array->data.array.items, capacity * sizeof(*items));
        if (items == NULL) return 0;
        array->data.array.items = items;
        array->data.array.capacity = capacity;
    }
    array->data.array.items[array->data.array.count++] = item;
    return 1;
}

static uint64_t hash_json_key(const char *key) {
    uint64_t hash = UINT64_C(1469598103934665603);
    for (const unsigned char *cursor = (const unsigned char *)key; *cursor; cursor++) {
        hash ^= *cursor;
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

static size_t object_slot(const JsonObject *object, const char *key) {
    size_t slot = (size_t)(hash_json_key(key) & (object->slot_count - 1));
    while (object->slots[slot] != 0) {
        const size_t index = object->slots[slot] - 1;
        if (strcmp(object->keys[index], key) == 0) return slot;
        slot = (slot + 1) & (object->slot_count - 1);
    }
    return slot;
}

static int resize_object_slots(JsonObject *object, size_t slot_count) {
    size_t *slots = calloc(slot_count, sizeof(*slots));
    if (slots == NULL) return 0;
    for (size_t index = 0; index < object->count; index++) {
        size_t slot = (size_t)(hash_json_key(object->keys[index]) & (slot_count - 1));
        while (slots[slot] != 0) slot = (slot + 1) & (slot_count - 1);
        slots[slot] = index + 1;
    }
    free(object->slots);
    object->slots = slots;
    object->slot_count = slot_count;
    return 1;
}

static int append_object(JsonValue *object, char *key, JsonValue *item) {
    JsonObject *entries = &object->data.object;
    if (entries->slot_count == 0 && !resize_object_slots(entries, 16)) return 0;
    size_t slot = object_slot(entries, key);
    if (entries->slots[slot] != 0) return 0;
    if (entries->count + 1 > entries->slot_count - entries->slot_count / 3) {
        if (entries->slot_count > SIZE_MAX / 2 ||
            !resize_object_slots(entries, entries->slot_count * 2)) return 0;
        slot = object_slot(entries, key);
    }
    if (entries->count == entries->capacity) {
        size_t capacity = entries->capacity == 0 ? 8 : entries->capacity * 2;
        if (capacity < entries->capacity || capacity > SIZE_MAX / sizeof(*entries->keys) ||
            capacity > SIZE_MAX / sizeof(*entries->values)) return 0;
        char **keys = malloc(capacity * sizeof(*keys));
        JsonValue **values = malloc(capacity * sizeof(*values));
        if (keys == NULL || values == NULL) {
            free(keys);
            free(values);
            return 0;
        }
        if (entries->count != 0) {
            memcpy(keys, entries->keys, entries->count * sizeof(*keys));
            memcpy(values, entries->values, entries->count * sizeof(*values));
        }
        free(entries->keys);
        free(entries->values);
        entries->keys = keys;
        entries->values = values;
        entries->capacity = capacity;
    }
    entries->keys[entries->count] = key;
    entries->values[entries->count] = item;
    entries->slots[slot] = entries->count + 1;
    entries->count++;
    return 1;
}

static int parse_number(JsonParser *parser, JsonValue *value) {
    const char *start = parser->cursor;
    if (parser->cursor < parser->end && *parser->cursor == '-') {
        parser->cursor++;
    }
    if (parser->cursor >= parser->end) {
        return 0;
    }
    if (*parser->cursor == '0') {
        parser->cursor++;
        if (parser->cursor < parser->end && isdigit((unsigned char)*parser->cursor)) {
            return 0;
        }
    } else if (*parser->cursor >= '1' && *parser->cursor <= '9') {
        while (parser->cursor < parser->end && isdigit((unsigned char)*parser->cursor)) {
            parser->cursor++;
        }
    } else {
        return 0;
    }
    if (parser->cursor < parser->end && *parser->cursor == '.') {
        parser->cursor++;
        if (parser->cursor >= parser->end || !isdigit((unsigned char)*parser->cursor)) {
            return 0;
        }
        while (parser->cursor < parser->end && isdigit((unsigned char)*parser->cursor)) {
            parser->cursor++;
        }
    }
    if (parser->cursor < parser->end && (*parser->cursor == 'e' || *parser->cursor == 'E')) {
        parser->cursor++;
        if (parser->cursor < parser->end && (*parser->cursor == '+' || *parser->cursor == '-')) {
            parser->cursor++;
        }
        if (parser->cursor >= parser->end || !isdigit((unsigned char)*parser->cursor)) {
            return 0;
        }
        while (parser->cursor < parser->end && isdigit((unsigned char)*parser->cursor)) {
            parser->cursor++;
        }
    }
    const size_t length = (size_t)(parser->cursor - start);
    char *text = malloc(length + 1);
    if (text == NULL) {
        set_error(parser, "out of memory parsing JSON number");
        return 0;
    }
    memcpy(text, start, length);
    text[length] = '\0';
    errno = 0;
    char *end = NULL;
    const double floating = strtod(text, &end);
    if (errno == ERANGE || end == text || *end != '\0' || !isfinite(floating)) {
        free(text);
        return 0;
    }
    value->data.number.floating = floating;
    value->data.number.text = text;
    return 1;
}

static JsonValue *parse_value(JsonParser *parser, size_t depth) {
    if (depth > JSON_MAX_DEPTH) {
        set_error(parser, "JSON nesting limit exceeded");
        return NULL;
    }
    skip_space(parser);
    if (parser->cursor >= parser->end) {
        return NULL;
    }
    JsonValue *value = NULL;
    if (*parser->cursor == '"') {
        value = new_value(JSON_STRING);
        if (value != NULL) {
            value->data.string = parse_string(parser);
            if (value->data.string == NULL) {
                json_free(value);
                value = NULL;
            }
        }
    } else if (*parser->cursor == '{') {
        value = new_value(JSON_OBJECT);
        if (value == NULL) {
            return NULL;
        }
        parser->cursor++;
        skip_space(parser);
        if (parser->cursor < parser->end && *parser->cursor == '}') {
            parser->cursor++;
            return value;
        }
        for (;;) {
            char *key = parse_string(parser);
            if (key == NULL) {
                json_free(value);
                return NULL;
            }
            skip_space(parser);
            if (parser->cursor >= parser->end || *parser->cursor++ != ':') {
                free(key);
                json_free(value);
                return NULL;
            }
            JsonValue *item = parse_value(parser, depth + 1);
            if (item == NULL || !append_object(value, key, item)) {
                free(key);
                json_free(item);
                json_free(value);
                return NULL;
            }
            skip_space(parser);
            if (parser->cursor >= parser->end) {
                json_free(value);
                return NULL;
            }
            const char delimiter = *parser->cursor++;
            if (delimiter == '}') {
                return value;
            }
            if (delimiter != ',') {
                json_free(value);
                return NULL;
            }
            skip_space(parser);
        }
    } else if (*parser->cursor == '[') {
        value = new_value(JSON_ARRAY);
        if (value == NULL) {
            return NULL;
        }
        parser->cursor++;
        skip_space(parser);
        if (parser->cursor < parser->end && *parser->cursor == ']') {
            parser->cursor++;
            return value;
        }
        for (;;) {
            JsonValue *item = parse_value(parser, depth + 1);
            if (item == NULL || !append_array(value, item)) {
                json_free(item);
                json_free(value);
                return NULL;
            }
            skip_space(parser);
            if (parser->cursor >= parser->end) {
                json_free(value);
                return NULL;
            }
            const char delimiter = *parser->cursor++;
            if (delimiter == ']') {
                return value;
            }
            if (delimiter != ',') {
                json_free(value);
                return NULL;
            }
        }
    } else if (parser->end - parser->cursor >= 4 &&
               strncmp(parser->cursor, "true", 4) == 0) {
        value = new_value(JSON_BOOLEAN);
        parser->cursor += 4;
        if (value != NULL) value->data.boolean = 1;
    } else if (parser->end - parser->cursor >= 5 &&
               strncmp(parser->cursor, "false", 5) == 0) {
        value = new_value(JSON_BOOLEAN);
        parser->cursor += 5;
    } else if (parser->end - parser->cursor >= 4 &&
               strncmp(parser->cursor, "null", 4) == 0) {
        value = new_value(JSON_NULL);
        parser->cursor += 4;
    } else {
        value = new_value(JSON_NUMBER);
        if (value == NULL || !parse_number(parser, value)) {
            json_free(value);
            return NULL;
        }
    }
    if (value == NULL) {
        set_error(parser, "out of memory parsing JSON value");
    }
    return value;
}

int json_parse(const char *text, JsonValue **result, char *error, size_t error_capacity) {
    if (error != NULL && error_capacity > 0) error[0] = '\0';
    if (text == NULL || result == NULL) {
        if (error != NULL && error_capacity > 0) snprintf(error, error_capacity, "invalid argument");
        return 0;
    }
    *result = NULL;
    JsonParser parser = {text, text + strlen(text), error, error_capacity};
    JsonValue *value = parse_value(&parser, 0);
    skip_space(&parser);
    if (value == NULL || parser.cursor != parser.end) {
        json_free(value);
        set_error(&parser, "invalid JSON document");
        return 0;
    }
    *result = value;
    return 1;
}

int json_parse_file(const char *path, JsonValue **result, char *error,
                    size_t error_capacity) {
    if (error != NULL && error_capacity > 0) error[0] = '\0';
    if (path == NULL || result == NULL) {
        if (error != NULL && error_capacity > 0) snprintf(error, error_capacity, "invalid argument");
        return 0;
    }
    *result = NULL;
    FILE *file = fopen(path, "rb");
    if (file == NULL) {
        if (error != NULL && error_capacity > 0) snprintf(error, error_capacity, "cannot open JSON file");
        return 0;
    }
    if (fseek(file, 0, SEEK_END) != 0) {
        fclose(file);
        if (error != NULL && error_capacity > 0) snprintf(error, error_capacity, "cannot seek JSON file");
        return 0;
    }
    const long size = ftell(file);
    if (size < 0 || (unsigned long)size > JSON_MAX_FILE_SIZE || fseek(file, 0, SEEK_SET) != 0) {
        fclose(file);
        if (error != NULL && error_capacity > 0) snprintf(error, error_capacity, "JSON file size is invalid");
        return 0;
    }
    char *text = malloc((size_t)size + 1);
    if (text == NULL) {
        fclose(file);
        if (error != NULL && error_capacity > 0) snprintf(error, error_capacity, "out of memory reading JSON file");
        return 0;
    }
    const size_t read = fread(text, 1, (size_t)size, file);
    const int close_status = fclose(file);
    const int read_ok = read == (size_t)size && close_status == 0;
    if (!read_ok) {
        free(text);
        if (error != NULL && error_capacity > 0) snprintf(error, error_capacity, "cannot read JSON file");
        return 0;
    }
    text[size] = '\0';
    const int parsed = json_parse(text, result, error, error_capacity);
    free(text);
    return parsed;
}

JsonType json_type(const JsonValue *value) {
    return value == NULL ? JSON_NULL : value->type;
}

const JsonValue *json_object_get(const JsonValue *object, const char *key) {
    if (object == NULL || object->type != JSON_OBJECT || key == NULL) {
        return NULL;
    }
    const JsonObject *entries = &object->data.object;
    if (entries->slot_count == 0) return NULL;
    const size_t slot = object_slot(entries, key);
    return entries->slots[slot] == 0 ? NULL : entries->values[entries->slots[slot] - 1];
}

size_t json_object_size(const JsonValue *object) {
    return object == NULL || object->type != JSON_OBJECT ? 0 : object->data.object.count;
}

const char *json_object_key_at(const JsonValue *object, size_t index) {
    if (object == NULL || object->type != JSON_OBJECT ||
        index >= object->data.object.count) {
        return NULL;
    }
    return object->data.object.keys[index];
}

const JsonValue *json_object_value_at(const JsonValue *object, size_t index) {
    if (object == NULL || object->type != JSON_OBJECT ||
        index >= object->data.object.count) {
        return NULL;
    }
    return object->data.object.values[index];
}

size_t json_array_size(const JsonValue *array) {
    return array == NULL || array->type != JSON_ARRAY ? 0 : array->data.array.count;
}

const JsonValue *json_array_get(const JsonValue *array, size_t index) {
    if (array == NULL || array->type != JSON_ARRAY || index >= array->data.array.count) {
        return NULL;
    }
    return array->data.array.items[index];
}

const char *json_string(const JsonValue *value) {
    return value == NULL || value->type != JSON_STRING ? NULL : value->data.string;
}

int json_number_double(const JsonValue *value, double *result) {
    if (value == NULL || result == NULL || value->type != JSON_NUMBER) {
        return 0;
    }
    *result = value->data.number.floating;
    return 1;
}

int json_number_u64(const JsonValue *value, uint64_t *result) {
    if (value == NULL || result == NULL || value->type != JSON_NUMBER ||
        value->data.number.text[0] == '-') {
        return 0;
    }
    for (const char *cursor = value->data.number.text; *cursor != '\0'; cursor++) {
        if (!isdigit((unsigned char)*cursor)) {
            return 0;
        }
    }
    errno = 0;
    char *end = NULL;
    const unsigned long long parsed = strtoull(value->data.number.text, &end, 10);
    if (errno == ERANGE || end == value->data.number.text || *end != '\0') {
        return 0;
    }
    *result = (uint64_t)parsed;
    return 1;
}

int json_boolean(const JsonValue *value, int *result) {
    if (value == NULL || result == NULL || value->type != JSON_BOOLEAN) {
        return 0;
    }
    *result = value->data.boolean;
    return 1;
}
