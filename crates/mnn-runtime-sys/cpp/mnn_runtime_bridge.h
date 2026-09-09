#ifndef MNN_RUNTIME_BRIDGE_H
#define MNN_RUNTIME_BRIDGE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct MnnRuntimeEngine MnnRuntimeEngine;

typedef struct MnnRuntimeConfig {
    int32_t backend;
    int32_t threads;
    int32_t precision;
    int32_t power;
    int32_t memory;
    int32_t gpu_mode;
    const char *cache_file;
} MnnRuntimeConfig;

enum MnnRuntimeStatus {
    MNN_RUNTIME_OK = 0,
    MNN_RUNTIME_INVALID_ARGUMENT = 1,
    MNN_RUNTIME_MODEL_ERROR = 2,
    MNN_RUNTIME_SESSION_ERROR = 3,
    MNN_RUNTIME_TENSOR_NOT_FOUND = 4,
    MNN_RUNTIME_TENSOR_TYPE_ERROR = 5,
    MNN_RUNTIME_SHAPE_ERROR = 6,
    MNN_RUNTIME_COPY_ERROR = 7,
    MNN_RUNTIME_INFERENCE_ERROR = 8,
    MNN_RUNTIME_INTERNAL_ERROR = 9
};

const char *mnn_runtime_version(void);
int32_t mnn_runtime_backend_available(int32_t backend);

MnnRuntimeEngine *mnn_runtime_engine_create(
    const void *model,
    size_t model_size,
    const MnnRuntimeConfig *config);
int32_t mnn_runtime_effective_threads(const MnnRuntimeEngine *engine);
int32_t mnn_runtime_tensor_layout(const MnnRuntimeEngine *engine, int32_t input, size_t index);
void mnn_runtime_engine_destroy(MnnRuntimeEngine *engine);
const char *mnn_runtime_last_error(const MnnRuntimeEngine *engine);

size_t mnn_runtime_tensor_count(const MnnRuntimeEngine *engine, int32_t input);
const char *mnn_runtime_tensor_name(
    const MnnRuntimeEngine *engine,
    int32_t input,
    size_t index);
size_t mnn_runtime_tensor_rank(
    const MnnRuntimeEngine *engine,
    int32_t input,
    size_t index);
int32_t mnn_runtime_tensor_shape(
    const MnnRuntimeEngine *engine,
    int32_t input,
    size_t index,
    int32_t *dimensions,
    size_t capacity);

typedef struct MnnRuntimeInputShape {
    size_t index;
    const int32_t *dimensions;
    size_t rank;
} MnnRuntimeInputShape;

/* Validate all input shapes before resizing any tensor. */
int32_t mnn_runtime_resize_inputs(
    MnnRuntimeEngine *engine,
    const MnnRuntimeInputShape *shapes,
    size_t count);
int32_t mnn_runtime_save_cache(MnnRuntimeEngine *engine);

int32_t mnn_runtime_write_input_index_f32(
    MnnRuntimeEngine *engine,
    size_t index,
    const float *data,
    size_t element_count);
int32_t mnn_runtime_run(MnnRuntimeEngine *engine);
int32_t mnn_runtime_read_output_index_f32(
    MnnRuntimeEngine *engine,
    size_t index,
    float *data,
    size_t element_count);

#ifdef __cplusplus
}
#endif

#endif

