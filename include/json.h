#ifndef JSON_H
#define JSON_H

#include <stddef.h>
#include <stdint.h>

typedef enum {
    JSON_NULL,
    JSON_BOOLEAN,
    JSON_NUMBER,
    JSON_STRING,
    JSON_ARRAY,
    JSON_OBJECT
} JsonType;

typedef struct JsonValue JsonValue;

int json_parse(const char *text, JsonValue **result, char *error, size_t error_capacity);
int json_parse_file(const char *path, JsonValue **result, char *error,
                    size_t error_capacity);
void json_free(JsonValue *value);
JsonType json_type(const JsonValue *value);
const JsonValue *json_object_get(const JsonValue *object, const char *key);
size_t json_object_size(const JsonValue *object);
const char *json_object_key_at(const JsonValue *object, size_t index);
const JsonValue *json_object_value_at(const JsonValue *object, size_t index);
size_t json_array_size(const JsonValue *array);
const JsonValue *json_array_get(const JsonValue *array, size_t index);
const char *json_string(const JsonValue *value);
int json_number_double(const JsonValue *value, double *result);
int json_number_u64(const JsonValue *value, uint64_t *result);
int json_boolean(const JsonValue *value, int *result);

#endif
