#ifndef LAYA_MODEL_H
#define LAYA_MODEL_H

#include <stddef.h>

typedef struct LayaModel LayaModel;

typedef struct {
    const char *name;
    const char *type;
    size_t option_count;
    const char *const *labels;
    float probabilities[3];
    size_t input_tokens;
} LayaDecision;

typedef struct {
    LayaDecision decisions[3];
} LayaPrediction;

int laya_model_open(const char *model_directory, LayaModel **result,
                    char *error, size_t error_capacity);
void laya_model_close(LayaModel *model);
int laya_model_predict(LayaModel *model, LayaPrediction *prediction,
                       char *error, size_t error_capacity);
const char *laya_model_backend(const LayaModel *model);

#endif
