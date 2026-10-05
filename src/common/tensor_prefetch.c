#include "tensor_prefetch.h"

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TENSOR_PREFETCH_ERROR_CAPACITY 256u

struct TensorPrefetch {
    const TensorStore *store;
    pthread_t thread;
    pthread_mutex_t mutex;
    pthread_cond_t condition;
    char *requested_name;
    TensorBuffer *loaded;
    char task_error[TENSOR_PREFETCH_ERROR_CAPACITY];
    int requested;
    int working;
    int completed;
    int stopping;
};

static void set_error(char *error, size_t capacity, const char *message) {
    if (error != NULL && capacity > 0) snprintf(error, capacity, "%s", message);
}

static char *copy_string(const char *value) {
    const size_t length = strlen(value);
    char *copy = malloc(length + 1);
    if (copy != NULL) memcpy(copy, value, length + 1);
    return copy;
}

static void *prefetch_worker(void *context) {
    TensorPrefetch *prefetch = context;
    for (;;) {
        pthread_mutex_lock(&prefetch->mutex);
        while (!prefetch->requested && !prefetch->stopping)
            pthread_cond_wait(&prefetch->condition, &prefetch->mutex);
        if (!prefetch->requested && prefetch->stopping) {
            pthread_mutex_unlock(&prefetch->mutex);
            return NULL;
        }
        char *tensor_name = prefetch->requested_name;
        prefetch->requested_name = NULL;
        prefetch->requested = 0;
        prefetch->working = 1;
        pthread_mutex_unlock(&prefetch->mutex);

        TensorBuffer *loaded = NULL;
        char task_error[TENSOR_PREFETCH_ERROR_CAPACITY] = {0};
        tensor_buffer_load(prefetch->store, tensor_name, &loaded, task_error,
                           sizeof(task_error));
        free(tensor_name);

        pthread_mutex_lock(&prefetch->mutex);
        prefetch->loaded = loaded;
        snprintf(prefetch->task_error, sizeof(prefetch->task_error), "%s",
                 task_error);
        prefetch->working = 0;
        prefetch->completed = 1;
        pthread_cond_broadcast(&prefetch->condition);
        pthread_mutex_unlock(&prefetch->mutex);
    }
}

int tensor_prefetch_create(const TensorStore *store, TensorPrefetch **result,
                           char *error, size_t error_capacity) {
    if (result != NULL) *result = NULL;
    if (store == NULL || result == NULL) {
        set_error(error, error_capacity, "prefetch requires a tensor store and result");
        return 0;
    }
    TensorPrefetch *prefetch = calloc(1, sizeof(*prefetch));
    if (prefetch == NULL) {
        set_error(error, error_capacity, "out of memory creating tensor prefetcher");
        return 0;
    }
    prefetch->store = store;
    if (pthread_mutex_init(&prefetch->mutex, NULL) != 0) {
        free(prefetch);
        set_error(error, error_capacity, "cannot initialize prefetch mutex");
        return 0;
    }
    if (pthread_cond_init(&prefetch->condition, NULL) != 0) {
        pthread_mutex_destroy(&prefetch->mutex);
        free(prefetch);
        set_error(error, error_capacity, "cannot initialize prefetch condition");
        return 0;
    }
    if (pthread_create(&prefetch->thread, NULL, prefetch_worker, prefetch) != 0) {
        pthread_cond_destroy(&prefetch->condition);
        pthread_mutex_destroy(&prefetch->mutex);
        free(prefetch);
        set_error(error, error_capacity, "cannot create prefetch worker thread");
        return 0;
    }
    *result = prefetch;
    return 1;
}

int tensor_prefetch_request(TensorPrefetch *prefetch, const char *tensor_name,
                            char *error, size_t error_capacity) {
    if (prefetch == NULL || tensor_name == NULL || tensor_name[0] == '\0') {
        set_error(error, error_capacity, "prefetch request requires a tensor name");
        return 0;
    }
    char *name_copy = copy_string(tensor_name);
    if (name_copy == NULL) {
        set_error(error, error_capacity, "out of memory copying prefetch tensor name");
        return 0;
    }
    pthread_mutex_lock(&prefetch->mutex);
    if (prefetch->stopping || prefetch->requested || prefetch->working ||
        prefetch->completed) {
        pthread_mutex_unlock(&prefetch->mutex);
        free(name_copy);
        set_error(error, error_capacity, "prefetcher is busy or stopping");
        return 0;
    }
    prefetch->requested_name = name_copy;
    prefetch->requested = 1;
    pthread_cond_signal(&prefetch->condition);
    pthread_mutex_unlock(&prefetch->mutex);
    return 1;
}

int tensor_prefetch_wait(TensorPrefetch *prefetch, TensorBuffer **result,
                         char *error, size_t error_capacity) {
    if (result != NULL) *result = NULL;
    if (prefetch == NULL || result == NULL) {
        set_error(error, error_capacity, "prefetch wait requires a result");
        return 0;
    }
    pthread_mutex_lock(&prefetch->mutex);
    if (!prefetch->requested && !prefetch->working && !prefetch->completed) {
        pthread_mutex_unlock(&prefetch->mutex);
        set_error(error, error_capacity, "no prefetch request is pending");
        return 0;
    }
    while (!prefetch->completed)
        pthread_cond_wait(&prefetch->condition, &prefetch->mutex);
    TensorBuffer *loaded = prefetch->loaded;
    prefetch->loaded = NULL;
    prefetch->completed = 0;
    if (loaded == NULL)
        set_error(error, error_capacity, prefetch->task_error);
    prefetch->task_error[0] = '\0';
    pthread_mutex_unlock(&prefetch->mutex);
    if (loaded == NULL) return 0;
    *result = loaded;
    return 1;
}

void tensor_prefetch_free(TensorPrefetch *prefetch) {
    if (prefetch == NULL) return;
    pthread_mutex_lock(&prefetch->mutex);
    prefetch->stopping = 1;
    pthread_cond_broadcast(&prefetch->condition);
    pthread_mutex_unlock(&prefetch->mutex);
    pthread_join(prefetch->thread, NULL);
    free(prefetch->requested_name);
    tensor_buffer_free(prefetch->loaded);
    pthread_cond_destroy(&prefetch->condition);
    pthread_mutex_destroy(&prefetch->mutex);
    free(prefetch);
}
