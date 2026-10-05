#include "smolvlm.h"
#include "safetensors.h"
#include "smolvlm_config.h"
#include "text_model.h"
#include "tokenizer.h"
#include "image.h"
#include "vision_model.h"
#include "timing.h"
#include "benchmark.h"
#include "tensor_compute_backend.h"

#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int smolvlm_print_version(void) {
    printf("smolvlm-c %s\n", SMOLVLM_C_VERSION);
    return 0;
}

int smolvlm_print_usage(const char *program_name) {
    fprintf(stderr, "Usage:\n  %s --version\n  %s inspect <model.safetensors>\n"
                    "  %s config <model-directory>\n"
                    "  %s vision-encode <model-directory> <square-tile-image>\n"
                    "  %s generate-image <model-directory> <image-path> [max-tokens] [prompt]\n"
                    "  %s generate <model-directory> <prompt> [max-new-tokens]\n"
                    "  %s benchmark <model-directory> <prompt> [max-new-tokens] [runs] [warmup-runs] [text|json]\n"
                    "  %s benchmark-vision <model-directory> <square-tile-image> [text|json]\n"
                    "  %s benchmark-image <model-directory> <image-path> <prompt> [max-new-tokens] [runs] [warmup-runs] [text|json]\n",
            program_name, program_name, program_name, program_name, program_name, program_name,
            program_name, program_name, program_name);
    return 2;
}

static char *join_path(const char *directory, const char *filename) {
    const size_t directory_length = strlen(directory);
    const size_t filename_length = strlen(filename);
    const int has_separator = directory_length > 0 &&
        directory[directory_length - 1] != '/' && directory[directory_length - 1] != '\\';
    if (directory_length > SIZE_MAX - filename_length - (size_t)has_separator - 1) {
        return NULL;
    }
    char *path = malloc(directory_length + filename_length + (size_t)has_separator + 1);
    if (path == NULL) {
        return NULL;
    }
    memcpy(path, directory, directory_length);
    size_t offset = directory_length;
    if (has_separator) path[offset++] = '/';
    memcpy(path + offset, filename, filename_length + 1);
    return path;
}

static int print_model_config(const char *directory) {
    char *config_path = join_path(directory, "config.json");
    char *generation_path = join_path(directory, "generation_config.json");
    if (config_path == NULL || generation_path == NULL) {
        free(config_path);
        free(generation_path);
        fprintf(stderr, "cannot allocate configuration paths\n");
        return 1;
    }
    SmolVLMConfig config;
    char error[256];
    const int loaded = smolvlm_config_load(config_path, generation_path, &config,
                                           error, sizeof(error));
    free(config_path);
    free(generation_path);
    if (!loaded) {
        fprintf(stderr, "%s\n", error);
        return 1;
    }
    printf("architecture=%s vocabulary=%zu image_token=", config.architecture,
           config.vocabulary_size);
    if (config.has_image_token_id) printf("%u", config.image_token_id);
    else printf("none");
    printf(" pixel_shuffle=%zu\n", config.pixel_shuffle_factor);
    printf("text: layers=%zu hidden=%zu intermediate=%zu q_heads=%zu kv_heads=%zu "
           "head_dim=%zu context=%zu rms_eps=%.8g rope_theta=%.8g\n",
           config.text.layers, config.text.hidden_size, config.text.intermediate_size,
           config.text.attention_heads, config.text.key_value_heads, config.text.head_dim,
           config.text.max_position_embeddings, config.text.rms_norm_eps,
           config.text.rope_theta);
    printf("vision: layers=%zu hidden=%zu heads=%zu image=%zu patch=%zu norm_eps=%.8g\n",
           config.vision.layers, config.vision.hidden_size, config.vision.attention_heads,
           config.vision.image_size, config.vision.patch_size, config.vision.layer_norm_eps);
    printf("generation: BOS=%u EOS=%u PAD=%u\n", config.bos_token_id,
           config.eos_token_id, config.pad_token_id);
    smolvlm_config_free(&config);
    return 0;
}

static int inspect_safetensors(const char *path) {
    SafeTensors *file = NULL;
    char error[256];
    if (!safetensors_open(path, &file, error, sizeof(error))) {
        fprintf(stderr, "%s: %s\n", path, error);
        return 1;
    }
    const size_t count = safetensors_count(file);
    printf("file=%s tensors=%zu\n", path, count);
    for (size_t index = 0; index < count; index++) {
        const SafeTensorInfo *tensor = safetensors_at(file, index);
        printf("%s %s [", tensor->name, tensor_dtype_name(tensor->dtype));
        for (size_t dimension = 0; dimension < tensor->rank; dimension++) {
            printf("%s%" PRIu64, dimension == 0 ? "" : ",", tensor->shape[dimension]);
        }
        printf("] bytes=%" PRIu64 " offset=%" PRIu64 "\n",
               tensor->byte_length, tensor->data_offset);
    }
    safetensors_close(file);
    return 0;
}

static int generate_text(const char *directory, const char *prompt,
                         const char *limit_text) {
    size_t max_new_tokens = 16;
    if (limit_text != NULL) {
        char *end = NULL;
        const unsigned long parsed = strtoul(limit_text, &end, 10);
        if (end == limit_text || *end != '\0' || parsed == 0 || parsed > 1024) {
            fprintf(stderr, "max-new-tokens must be between 1 and 1024\n");
            return 2;
        }
        max_new_tokens = (size_t)parsed;
    }
    char *tokenizer_path = join_path(directory, "tokenizer.json");
    if (tokenizer_path == NULL) {
        fprintf(stderr, "cannot allocate tokenizer path\n");
        return 1;
    }
    char error[512] = {0};
    Tokenizer *tokenizer = NULL;
    TextModel *model = NULL;
    char *generated = NULL;
    int ok = tokenizer_open(tokenizer_path, &tokenizer, error, sizeof(error));
    free(tokenizer_path);
    if (ok) ok = text_model_open(directory, &model, error, sizeof(error));
    if (ok) ok = text_model_generate(model, tokenizer, prompt, max_new_tokens,
                                     &generated, error, sizeof(error));
    if (!ok) fprintf(stderr, "%s\n", error);
    else printf("%s\n", generated);
    free(generated);
    text_model_close(model);
    tokenizer_close(tokenizer);
    return ok ? 0 : 1;
}

static int encode_vision_tile(const char *directory, const char *image_path) {
    char error[512] = {0};
    VisionModel *model = NULL;
    RgbImage image = {0};
    float *pixels = NULL;
    float *features = NULL;
    int ok = vision_model_open(directory, &model, error, sizeof(error));
    if (ok) ok = image_load(image_path, &image, error, sizeof(error));
    if (ok && (image.width != vision_model_input_size(model) ||
               image.height != vision_model_input_size(model))) {
        fprintf(stderr, "image must be a square %zu x %zu tile\n",
                vision_model_input_size(model), vision_model_input_size(model));
        ok = 0;
    }
    const float mean[] = {0.5f, 0.5f, 0.5f};
    const float deviation[] = {0.5f, 0.5f, 0.5f};
    if (ok) ok = image_resize_normalize(&image, image.width, image.height,
        mean, deviation, &pixels, error, sizeof(error));
    if (ok) ok = vision_model_encode_tile(model, pixels, &features, error, sizeof(error));
    if (ok) {
        const size_t count = vision_model_output_tokens(model) * vision_model_output_width(model);
        double checksum = 0.0;
        for (size_t i = 0; i < count; i++) checksum += features[i];
        printf("vision_features=[%zu,%zu] checksum=%.9g\n",
               vision_model_output_tokens(model), vision_model_output_width(model), checksum);
    } else if (error[0] != '\0') {
        fprintf(stderr, "%s\n", error);
    }
    free(features);
    free(pixels);
    image_free(&image);
    vision_model_close(model);
    return ok ? 0 : 1;
}

static int generate_image(const char *directory, const char *image_path,
                         int optional_count, char **optional_args) {
    static const char default_prompt[] =
        "Extract the visible market data. Return only JSON with keys \"instrument_name\", \"symbol\", \"snapshot_time\", \"last_price\", \"change\", \"change_pct\", \"best_ask\", \"best_ask_qty\", \"best_bid\", \"best_bid_qty\", \"high\", \"low\", \"asks\", \"bids\", and \"trades\". symbol is the contract code; instrument_name is the displayed Chinese contract name. snapshot_time is the latest visible trade time in HH:MM:SS format. best_ask/best_bid and their quantities come from level 1 of the book. last_price, change, change_pct, high, and low are numbers. Preserve signs; express change_pct in percentage points (for example -3.06 for -3.06%). asks and bids are lists of [price, qty] arrays, ordered from level 1 to 5. trades is a list of [time, price, qty, oi_change, open_close] arrays. time is HH:MM:SS; price is numeric; qty and oi_change are integers. oi_change is the signed position change; open_close is the displayed Chinese trade classification. Read trade rows from top to bottom, preserving that order even for equal timestamps. Do not include any other fields or invent missing values.";
    size_t max_tokens = 2048;
    int first_prompt = 0;
    if (optional_count > 0) {
        char *end = NULL;
        const unsigned long parsed = strtoul(optional_args[0], &end, 10);
        if (end != optional_args[0] && *end == '\0') {
            if (parsed == 0 || parsed > 4096) {
                fprintf(stderr, "max-tokens must be between 1 and 4096\n");
                return 2;
            }
            max_tokens = (size_t)parsed;
            first_prompt = 1;
        }
    }
    size_t prompt_size = 1;
    for (int i = first_prompt; i < optional_count; i++) {
        const size_t length = strlen(optional_args[i]);
        const size_t separator = i > first_prompt ? 1 : 0;
        if (length > SIZE_MAX - prompt_size - separator) return 1;
        prompt_size += length + separator;
    }
    char *prompt = optional_count == first_prompt ? NULL : malloc(prompt_size);
    if (optional_count != first_prompt && prompt == NULL) {
        fprintf(stderr, "out of memory formatting image prompt\n");
        return 1;
    }
    if (prompt != NULL) {
        prompt[0] = '\0';
        for (int i = first_prompt; i < optional_count; i++) {
            if (i > first_prompt) strcat(prompt, " ");
            strcat(prompt, optional_args[i]);
        }
    } else prompt = (char *)default_prompt;

    char error[512] = {0};
    char *tokenizer_path = join_path(directory, "tokenizer.json");
    Tokenizer *tokenizer = NULL;
    TextModel *text = NULL;
    VisionModel *vision = NULL;
    RgbImage image = {0};
    ImageTileSet tiles = {0};
    float *all_features = NULL;
    char *completion = NULL;
    double phase_started = timing_start();
    int ok = tokenizer_path != NULL && tokenizer_open(tokenizer_path, &tokenizer, error, sizeof(error));
    timing_report("tokenizer_load", phase_started);
    free(tokenizer_path);
    phase_started = timing_start();
    if (ok) ok = image_load(image_path, &image, error, sizeof(error));
    timing_report("image_decode", phase_started);
    phase_started = timing_start();
    if (ok) ok = vision_model_open(directory, &vision, error, sizeof(error));
    timing_report("vision_model_load", phase_started);
    const float mean[] = {0.5f, 0.5f, 0.5f};
    const float deviation[] = {0.5f, 0.5f, 0.5f};
    const size_t edge = vision_model_input_size(vision);
    phase_started = timing_start();
    if (ok) ok = image_preprocess_tiles(&image, edge, edge * 4, mean, deviation,
                                         &tiles, error, sizeof(error));
    timing_report("image_preprocess", phase_started);
    const size_t per_tile_tokens = vision_model_output_tokens(vision);
    const size_t feature_width = vision_model_output_width(vision);
    if (ok && (per_tile_tokens == 0 || feature_width == 0 ||
        tiles.count > SIZE_MAX / per_tile_tokens ||
        tiles.count * per_tile_tokens > SIZE_MAX / feature_width / sizeof(float))) {
        snprintf(error, sizeof(error), "image feature dimensions overflow");
        ok = 0;
    }
    const size_t image_token_count = ok ? tiles.count * per_tile_tokens : 0;
    if (ok) all_features = malloc(image_token_count * feature_width * sizeof(float));
    if (ok && all_features == NULL) {
        snprintf(error, sizeof(error), "out of memory allocating image embeddings");
        ok = 0;
    }
    phase_started = timing_start();
    for (size_t tile = 0; ok && tile < tiles.count; tile++) {
        float *tile_features = NULL;
        ok = vision_model_encode_tile(vision, tiles.normalized_chw[tile],
                                      &tile_features, error, sizeof(error));
        if (ok) memcpy(all_features + tile * per_tile_tokens * feature_width,
                       tile_features, per_tile_tokens * feature_width * sizeof(float));
        free(tile_features);
    }
    timing_report("vision_encode", phase_started);
    vision_model_close(vision);
    vision = NULL;
    phase_started = timing_start();
    if (ok) ok = text_model_open(directory, &text, error, sizeof(error));
    timing_report("text_model_load", phase_started);
    phase_started = timing_start();
    if (ok) ok = text_model_generate_image(text, tokenizer, prompt, max_tokens,
        tiles.rows, tiles.columns, per_tile_tokens, all_features, image_token_count,
        &completion, error, sizeof(error));
    timing_report("image_generation_total", phase_started);
    if (ok) printf("%s\n", completion);
    else if (error[0] != '\0') fprintf(stderr, "%s\n", error);
    free(completion);
    free(all_features);
    image_tile_set_free(&tiles);
    image_free(&image);
    vision_model_close(vision);
    text_model_close(text);
    tokenizer_close(tokenizer);
    if (prompt != default_prompt) free(prompt);
    return ok ? 0 : 1;
}

typedef struct {
    const char *model_directory;
    const char *image_path;
    VisionModel *model;
    RgbImage image;
    float *pixels;
} VisionBenchmarkWorkload;

static int vision_benchmark_prepare(void *context, char *error, size_t capacity) {
    VisionBenchmarkWorkload *workload = context;
    const TensorComputeBackend *backend = tensor_compute_preferred_backend();
    if (backend == NULL || backend->name == NULL ||
        (strcmp(backend->name, "metal") != 0 && strcmp(backend->name, "cuda") != 0)) {
        snprintf(error, capacity, "vision benchmark requires an available Metal or CUDA backend");
        return 0;
    }
    int ok = vision_model_open(workload->model_directory, &workload->model,
                               error, capacity);
    if (ok && !vision_model_uses_fused_sequence(workload->model)) {
        snprintf(error, capacity,
                 "optimized fused vision sequence path is unavailable for this model/backend");
        ok = 0;
    }
    if (ok) ok = image_load(workload->image_path, &workload->image, error, capacity);
    const float mean[] = {0.5f, 0.5f, 0.5f};
    const float deviation[] = {0.5f, 0.5f, 0.5f};
    if (ok) ok = image_resize_normalize(&workload->image,
        vision_model_input_size(workload->model),
        vision_model_input_size(workload->model), mean, deviation,
        &workload->pixels, error, capacity);
    if (!ok) {
        image_free(&workload->image);
        vision_model_close(workload->model);
        workload->model = NULL;
    }
    return ok;
}

static int vision_benchmark_run(void *context, double *work_units,
                                char *error, size_t capacity) {
    VisionBenchmarkWorkload *workload = context;
    float *features = NULL;
    const int ok = vision_model_encode_tile(workload->model, workload->pixels,
                                             &features, error, capacity);
    free(features);
    if (ok) *work_units = 1.0;
    return ok;
}

static void vision_benchmark_cleanup(void *context) {
    VisionBenchmarkWorkload *workload = context;
    free(workload->pixels);
    image_free(&workload->image);
    vision_model_close(workload->model);
    workload->pixels = NULL;
    workload->model = NULL;
}

typedef struct {
    const char *model_directory;
    const char *image_path;
    const char *prompt;
    size_t max_new_tokens;
    size_t warmup_runs_remaining;
    size_t image_tokens_per_tile;
    size_t image_token_count;
    size_t feature_width;
    size_t image_rows;
    size_t image_columns;
    double measured_decode_ms;
    double measured_decode_tokens;
    Tokenizer *tokenizer;
    TextModel *text_model;
    VisionModel *vision_model;
    RgbImage image;
    ImageTileSet tiles;
    float *all_features;
} ImageGenerationBenchmarkWorkload;

static int image_benchmark_prepare(void *context, char *error, size_t capacity) {
    ImageGenerationBenchmarkWorkload *workload = context;
    const TensorComputeBackend *backend = tensor_compute_preferred_backend();
    if (backend == NULL || backend->name == NULL ||
        (strcmp(backend->name, "metal") != 0 && strcmp(backend->name, "cuda") != 0)) {
        snprintf(error, capacity, "vision benchmark requires an available Metal or CUDA backend");
        return 0;
    }
    char *tokenizer_path = join_path(workload->model_directory, "tokenizer.json");
    if (tokenizer_path == NULL) {
        snprintf(error, capacity, "cannot allocate tokenizer path");
        return 0;
    }
    int ok = tokenizer_open(tokenizer_path, &workload->tokenizer, error, capacity);
    free(tokenizer_path);
    if (ok) ok = text_model_open(workload->model_directory, &workload->text_model,
                                 error, capacity);
    if (ok) ok = vision_model_open(workload->model_directory, &workload->vision_model,
                                   error, capacity);
    if (ok && !vision_model_uses_fused_sequence(workload->vision_model)) {
        snprintf(error, capacity,
                 "optimized fused vision sequence path is unavailable for this model/backend");
        ok = 0;
    }
    if (ok) ok = image_load(workload->image_path, &workload->image, error, capacity);
    const float mean[] = {0.5f, 0.5f, 0.5f};
    const float deviation[] = {0.5f, 0.5f, 0.5f};
    const size_t edge = ok ? vision_model_input_size(workload->vision_model) : 0;
    if (ok) ok = image_preprocess_tiles(&workload->image, edge, edge * 4,
        mean, deviation, &workload->tiles, error, capacity);
    if (ok) {
        workload->image_tokens_per_tile = vision_model_output_tokens(workload->vision_model);
        workload->feature_width = vision_model_output_width(workload->vision_model);
        if (workload->image_tokens_per_tile == 0 || workload->feature_width == 0 ||
            workload->tiles.count > SIZE_MAX / workload->image_tokens_per_tile ||
            workload->tiles.count * workload->image_tokens_per_tile >
                SIZE_MAX / workload->feature_width / sizeof(float)) {
            snprintf(error, capacity, "image feature dimensions overflow");
            ok = 0;
        } else {
            workload->image_token_count = workload->tiles.count * workload->image_tokens_per_tile;
            workload->image_rows = workload->tiles.rows;
            workload->image_columns = workload->tiles.columns;
            workload->all_features = malloc(workload->image_token_count *
                workload->feature_width * sizeof(float));
            if (workload->all_features == NULL) {
                snprintf(error, capacity, "out of memory allocating image embeddings");
                ok = 0;
            }
        }
    }
    if (!ok) {
        free(workload->all_features);
        workload->all_features = NULL;
        image_tile_set_free(&workload->tiles);
        image_free(&workload->image);
        vision_model_close(workload->vision_model);
        workload->vision_model = NULL;
        text_model_close(workload->text_model);
        workload->text_model = NULL;
        tokenizer_close(workload->tokenizer);
        workload->tokenizer = NULL;
    }
    return ok;
}

static int image_benchmark_run(void *context, double *work_units,
                                char *error, size_t capacity) {
    ImageGenerationBenchmarkWorkload *workload = context;
    int ok = 1;
    for (size_t tile = 0; ok && tile < workload->tiles.count; tile++) {
        float *features = NULL;
        ok = vision_model_encode_tile(workload->vision_model,
            workload->tiles.normalized_chw[tile], &features, error, capacity);
        if (ok) memcpy(workload->all_features + tile * workload->image_tokens_per_tile *
            workload->feature_width, features, workload->image_tokens_per_tile *
            workload->feature_width * sizeof(float));
        free(features);
    }
    char *completion = NULL;
    TextGenerationStats stats = {0};
    if (ok) ok = text_model_generate_image_with_stats(workload->text_model,
        workload->tokenizer, workload->prompt, workload->max_new_tokens,
        workload->image_rows, workload->image_columns,
        workload->image_tokens_per_tile, workload->all_features,
        workload->image_token_count, &completion, &stats, error, capacity);
    free(completion);
    if (ok && !stats.used_accelerated_decode) {
        snprintf(error, capacity,
                 "accelerated Metal/CUDA decode path is unavailable; refusing CPU benchmark");
        ok = 0;
    }
    if (ok) {
        *work_units = (double)stats.generated_tokens;
        if (workload->warmup_runs_remaining > 0) workload->warmup_runs_remaining--;
        else {
            workload->measured_decode_ms += stats.decode_latency_ms;
            workload->measured_decode_tokens += (double)stats.generated_tokens;
        }
    }
    return ok;
}

static void image_benchmark_cleanup(void *context) {
    ImageGenerationBenchmarkWorkload *workload = context;
    free(workload->all_features);
    image_tile_set_free(&workload->tiles);
    image_free(&workload->image);
    vision_model_close(workload->vision_model);
    text_model_close(workload->text_model);
    tokenizer_close(workload->tokenizer);
    workload->all_features = NULL;
    workload->vision_model = NULL;
    workload->text_model = NULL;
    workload->tokenizer = NULL;
}

static int benchmark_vision_command(int argc, char **argv) {
    if (argc < 4 || argc > 5) return smolvlm_print_usage(argv[0]);
    const char *format = argc == 5 ? argv[4] : "text";
    if (strcmp(format, "text") != 0 && strcmp(format, "json") != 0)
        return smolvlm_print_usage(argv[0]);
    VisionBenchmarkWorkload context = {
        .model_directory = argv[2], .image_path = argv[3]
    };
    const BenchmarkWorkload workload = {
        .name = "vision_encoder_fused", .unit = "tile", .context = &context,
        .prepare = vision_benchmark_prepare, .run = vision_benchmark_run,
        .cleanup = vision_benchmark_cleanup
    };
#if defined(__APPLE__)
    const char *platform = "macOS";
#elif defined(__linux__)
    const char *platform = "Linux";
#else
    const char *platform = "unknown";
#endif
    const TensorComputeBackend *compute = tensor_compute_preferred_backend();
    const BenchmarkConfig config = {
        .warmup_iterations = 2, .measured_iterations = 10,
        .platform = platform,
        .backend = compute != NULL && compute->name != NULL ? compute->name : "unavailable",
        .device = "selected-gpu"
    };
    char error[512] = {0};
    BenchmarkSession *session = NULL;
    int ok = benchmark_session_create(&config, &workload, &session,
                                      error, sizeof(error));
    if (ok) ok = benchmark_session_run(session, error, sizeof(error));
    char *report_output = NULL;
    if (ok && strcmp(format, "json") == 0)
        ok = benchmark_report_to_json(benchmark_session_report(session),
                                      &report_output, error, sizeof(error));
    else if (ok)
        ok = benchmark_report_to_text(benchmark_session_report(session),
                                      &report_output, error, sizeof(error));
    if (ok) puts(report_output);
    else fprintf(stderr, "%s\n", error[0] == '\0' ? "vision benchmark failed" : error);
    free(report_output);
    benchmark_session_destroy(session);
    return ok ? 0 : 1;
}

static int parse_benchmark_count(const char *text, size_t fallback,
                                 size_t maximum, size_t *result);

static int benchmark_image_command(int argc, char **argv) {
    if (argc < 5 || argc > 9) return smolvlm_print_usage(argv[0]);
    const char *format = argc > 8 ? argv[8] : "text";
    if (strcmp(format, "text") != 0 && strcmp(format, "json") != 0)
        return smolvlm_print_usage(argv[0]);
    size_t max_new_tokens, measured_iterations, warmup_iterations = 0;
    if (!parse_benchmark_count(argc > 5 ? argv[5] : NULL, 1024, 1024, &max_new_tokens) ||
        !parse_benchmark_count(argc > 6 ? argv[6] : NULL, 1, 1000, &measured_iterations)) {
        fprintf(stderr, "benchmark-image limits: max-new-tokens 1..1024, runs 1..1000, warmup-runs 0..1000\n");
        return 2;
    }
    if (argc > 7) {
        char *end = NULL;
        const unsigned long parsed = strtoul(argv[7], &end, 10);
        if (end == argv[7] || *end != '\0' || parsed > 1000) {
            fprintf(stderr, "benchmark-image warmup-runs must be between 0 and 1000\n");
            return 2;
        }
        warmup_iterations = (size_t)parsed;
    }
    ImageGenerationBenchmarkWorkload context = {
        .model_directory = argv[2], .image_path = argv[3], .prompt = argv[4],
        .max_new_tokens = max_new_tokens,
        .warmup_runs_remaining = warmup_iterations
    };
    const BenchmarkWorkload workload = {
        .name = "image_text_generation_accelerated_decode", .unit = "token",
        .context = &context, .prepare = image_benchmark_prepare,
        .run = image_benchmark_run, .cleanup = image_benchmark_cleanup
    };
    const TensorComputeBackend *compute = tensor_compute_preferred_backend();
    if (compute == NULL || compute->name == NULL ||
        (strcmp(compute->name, "metal") != 0 && strcmp(compute->name, "cuda") != 0)) {
        fprintf(stderr, "benchmark-image requires an available Metal or CUDA compute backend\n");
        return 1;
    }
#if defined(__APPLE__)
    const char *platform = "macOS";
#elif defined(__linux__)
    const char *platform = "Linux";
#else
    const char *platform = "unknown";
#endif
    const BenchmarkConfig config = {
        .warmup_iterations = warmup_iterations,
        .measured_iterations = measured_iterations,
        .platform = platform, .backend = compute->name, .device = "selected-gpu"
    };
    char error[512] = {0};
    BenchmarkSession *session = NULL;
    int ok = benchmark_session_create(&config, &workload, &session, error, sizeof(error));
    if (ok) ok = benchmark_session_run(session, error, sizeof(error));
    char *report_output = NULL;
    if (ok && strcmp(format, "json") == 0)
        ok = benchmark_report_to_json(benchmark_session_report(session),
                                      &report_output, error, sizeof(error));
    else if (ok)
        ok = benchmark_report_to_text(benchmark_session_report(session),
                                      &report_output, error, sizeof(error));
    if (ok) {
        const double decode_tps = context.measured_decode_ms > 0.0 ?
            context.measured_decode_tokens * 1000.0 / context.measured_decode_ms : 0.0;
        if (strcmp(format, "json") == 0) {
            const size_t output_length = strlen(report_output);
            fwrite(report_output, 1, output_length - 1, stdout);
            printf(",\"optimized_decode_path\":true,\"decode_generated_tokens\":%.0f"
                   ",\"decode_total_latency_ms\":%.9g,\"decode_tokens_per_second\":%.9g}\n",
                   context.measured_decode_tokens, context.measured_decode_ms, decode_tps);
        } else {
            puts(report_output);
            printf("  accelerated decode: %.0f tokens in %.1fms → %.1f tok/s\n",
                   context.measured_decode_tokens, context.measured_decode_ms, decode_tps);
        }
    } else fprintf(stderr, "%s\n", error[0] == '\0' ? "image benchmark failed" : error);
    free(report_output);
    benchmark_session_destroy(session);
    return ok ? 0 : 1;
}

typedef struct {
    const char *model_directory;
    const char *prompt;
    size_t max_new_tokens;
    size_t warmup_runs_remaining;
    double measured_decode_ms;
    double measured_decode_tokens;
    Tokenizer *tokenizer;
    TextModel *model;
} TextBenchmarkWorkload;

static int text_benchmark_prepare(void *context, char *error, size_t capacity) {
    TextBenchmarkWorkload *workload = context;
    char *tokenizer_path = join_path(workload->model_directory, "tokenizer.json");
    if (tokenizer_path == NULL) {
        snprintf(error, capacity, "cannot allocate tokenizer path");
        return 0;
    }
    int ok = tokenizer_open(tokenizer_path, &workload->tokenizer, error, capacity);
    free(tokenizer_path);
    if (ok) ok = text_model_open(workload->model_directory, &workload->model,
                                 error, capacity);
    if (!ok) {
        text_model_close(workload->model);
        workload->model = NULL;
        tokenizer_close(workload->tokenizer);
        workload->tokenizer = NULL;
    }
    return ok;
}

static int text_benchmark_run(void *context, double *work_units,
                              char *error, size_t capacity) {
    TextBenchmarkWorkload *workload = context;
    char *completion = NULL;
    TextGenerationStats stats = {0};
    int ok = text_model_generate_with_stats(workload->model, workload->tokenizer,
        workload->prompt, workload->max_new_tokens, &completion, &stats,
        error, capacity);
    free(completion);
    if (ok && !stats.used_accelerated_decode) {
        snprintf(error, capacity,
                 "accelerated Metal/CUDA decode path is unavailable; refusing CPU benchmark");
        ok = 0;
    }
    if (ok) {
        *work_units = (double)stats.generated_tokens;
        if (workload->warmup_runs_remaining > 0) {
            workload->warmup_runs_remaining--;
        } else {
            workload->measured_decode_ms += stats.decode_latency_ms;
            workload->measured_decode_tokens += (double)stats.generated_tokens;
        }
    }
    return ok;
}

static void text_benchmark_cleanup(void *context) {
    TextBenchmarkWorkload *workload = context;
    text_model_close(workload->model);
    tokenizer_close(workload->tokenizer);
    workload->model = NULL;
    workload->tokenizer = NULL;
}

static int parse_benchmark_count(const char *text, size_t fallback,
                                 size_t maximum, size_t *result) {
    if (text == NULL) { *result = fallback; return 1; }
    char *end = NULL;
    const unsigned long parsed = strtoul(text, &end, 10);
    if (end == text || *end != '\0' || parsed == 0 || parsed > maximum) return 0;
    *result = (size_t)parsed;
    return 1;
}

static int benchmark_text_command(int argc, char **argv) {
    if (argc < 4 || argc > 8) return smolvlm_print_usage(argv[0]);
    const char *format = argc > 7 ? argv[7] : "text";
    if (strcmp(format, "text") != 0 && strcmp(format, "json") != 0)
        return smolvlm_print_usage(argv[0]);
    size_t max_new_tokens, measured_iterations, warmup_iterations;
    if (!parse_benchmark_count(argc > 4 ? argv[4] : NULL, 16, 1024, &max_new_tokens) ||
        !parse_benchmark_count(argc > 5 ? argv[5] : NULL, 10, 1000, &measured_iterations) ||
        !parse_benchmark_count(argc > 6 ? argv[6] : NULL, 2, 1000, &warmup_iterations)) {
        fprintf(stderr, "benchmark limits: max-new-tokens 1..1024, runs 1..1000, warmup-runs 1..1000\n");
        return 2;
    }
    TextBenchmarkWorkload context = {
        .model_directory = argv[2], .prompt = argv[3],
        .max_new_tokens = max_new_tokens,
        .warmup_runs_remaining = warmup_iterations
    };
    const BenchmarkWorkload workload = {
        .name = "text_generation_accelerated_decode", .unit = "token",
        .context = &context,
        .prepare = text_benchmark_prepare, .run = text_benchmark_run,
        .cleanup = text_benchmark_cleanup
    };
    const TensorComputeBackend *compute = tensor_compute_preferred_backend();
    if (compute == NULL || compute->name == NULL ||
        (strcmp(compute->name, "metal") != 0 && strcmp(compute->name, "cuda") != 0)) {
        fprintf(stderr, "benchmark requires an available Metal or CUDA compute backend\n");
        return 1;
    }
#if defined(__APPLE__)
    const char *platform = "macOS";
#elif defined(__linux__)
    const char *platform = "Linux";
#else
    const char *platform = "unknown";
#endif
    const BenchmarkConfig config = {
        .warmup_iterations = warmup_iterations,
        .measured_iterations = measured_iterations,
        .platform = platform,
        .backend = compute != NULL && compute->name != NULL ? compute->name : "cpu",
        .device = "selected-gpu"
    };
    char error[512] = {0};
    BenchmarkSession *session = NULL;
    int ok = benchmark_session_create(&config, &workload, &session,
                                      error, sizeof(error));
    if (ok) ok = benchmark_session_run(session, error, sizeof(error));
    char *report_output = NULL;
    if (ok && strcmp(format, "json") == 0)
        ok = benchmark_report_to_json(benchmark_session_report(session),
                                      &report_output, error, sizeof(error));
    else if (ok)
        ok = benchmark_report_to_text(benchmark_session_report(session),
                                      &report_output, error, sizeof(error));
    if (ok) {
        const double mean_decode_ms = context.measured_decode_ms /
            (double)measured_iterations;
        const double decode_tokens_per_second = context.measured_decode_ms > 0.0 ?
            context.measured_decode_tokens * 1000.0 / context.measured_decode_ms : 0.0;
        if (strcmp(format, "json") == 0) {
            const size_t output_length = strlen(report_output);
            fwrite(report_output, 1, output_length - 1, stdout);
            printf(",\"optimized_decode_path\":true,\"decode_generated_tokens\":%.0f"
                   ",\"decode_total_latency_ms\":%.9g,\"decode_mean_latency_ms\":%.9g"
                   ",\"decode_tokens_per_second\":%.9g}\n",
                   context.measured_decode_tokens, context.measured_decode_ms,
                   mean_decode_ms, decode_tokens_per_second);
        } else {
            puts(report_output);
            printf("  accelerated decode: %.0f tokens in %.1fms → %.1f tok/s\n",
                   context.measured_decode_tokens, context.measured_decode_ms,
                   decode_tokens_per_second);
        }
    }
    else fprintf(stderr, "%s\n", error[0] == '\0' ? "benchmark failed" : error);
    free(report_output);
    benchmark_session_destroy(session);
    return ok ? 0 : 1;
}

int main(int argc, char **argv) {
    if (argc == 2 && strcmp(argv[1], "--version") == 0) {
        return smolvlm_print_version();
    }
    if (argc == 3 && strcmp(argv[1], "inspect") == 0) {
        return inspect_safetensors(argv[2]);
    }
    if (argc == 3 && strcmp(argv[1], "config") == 0) {
        return print_model_config(argv[2]);
    }
    if (argc == 4 && strcmp(argv[1], "vision-encode") == 0) {
        return encode_vision_tile(argv[2], argv[3]);
    }
    if (argc >= 4 && strcmp(argv[1], "generate-image") == 0) {
        return generate_image(argv[2], argv[3], argc - 4, argv + 4);
    }
    if ((argc == 4 || argc == 5) && strcmp(argv[1], "generate") == 0) {
        return generate_text(argv[2], argv[3], argc == 5 ? argv[4] : NULL);
    }
    if (argc >= 2 && strcmp(argv[1], "benchmark") == 0) {
        return benchmark_text_command(argc, argv);
    }
    if (argc == 4 && strcmp(argv[1], "benchmark-vision") == 0) {
        return benchmark_vision_command(argc, argv);
    }
    if (argc >= 5 && strcmp(argv[1], "benchmark-image") == 0) {
        return benchmark_image_command(argc, argv);
    }
    return smolvlm_print_usage(argv[0]);
}
