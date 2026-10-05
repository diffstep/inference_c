#include "json.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>

int main(void) {
    const char *text =
        "{\"text\":\"hello \\u4e16\\u754c\",\"items\":[1,true,null],"
        "\"large\":18446744073709551615,\"fraction\":1.25e2}";
    JsonValue *root = NULL;
    char error[128];
    assert(json_parse(text, &root, error, sizeof(error)));

    const char *decoded = json_string(json_object_get(root, "text"));
    assert(decoded != NULL && strcmp(decoded, "hello 世界") == 0);
    const JsonValue *items = json_object_get(root, "items");
    assert(json_array_size(items) == 3);
    uint64_t integer = 0;
    assert(json_number_u64(json_array_get(items, 0), &integer) && integer == 1);
    int boolean = 0;
    assert(json_boolean(json_array_get(items, 1), &boolean) && boolean == 1);
    assert(json_type(json_array_get(items, 2)) == JSON_NULL);
    assert(json_number_u64(json_object_get(root, "large"), &integer));
    assert(integer == UINT64_MAX);
    double fraction = 0;
    assert(json_number_double(json_object_get(root, "fraction"), &fraction));
    assert(fraction == 125.0);
    json_free(root);

    root = NULL;
    assert(!json_parse("{\"duplicate\":1,\"duplicate\":2}", &root,
                       error, sizeof(error)));
    assert(root == NULL);
    assert(!json_parse("[01]", &root, error, sizeof(error)));
    puts("json tests passed");
    return 0;
}
