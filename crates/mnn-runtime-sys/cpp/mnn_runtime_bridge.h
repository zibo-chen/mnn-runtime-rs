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

int32_t mnn_runtime_write_input_f32(
    MnnRuntimeEngine *engine,
    const uint8_t *name,
    size_t name_length,
    const float *data,
    size_t element_count);
int32_t mnn_runtime_run(MnnRuntimeEngine *engine);
int32_t mnn_runtime_read_output_f32(
    MnnRuntimeEngine *engine,
    const uint8_t *name,
    size_t name_length,
    float *data,
    size_t element_count);

#ifdef __cplusplus
}
#endif

#endif

