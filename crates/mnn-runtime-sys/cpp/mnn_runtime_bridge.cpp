#include "mnn_runtime_bridge.h"

#include <MNN/Interpreter.hpp>
#include <MNN/MNNDefine.h>
#include <MNN/Tensor.hpp>

#include <algorithm>
#include <cstring>
#include <exception>
#include <map>
#include <limits>
#include <memory>
#include <new>
#include <string>
#include <utility>
#include <vector>

namespace MNN {
class RuntimeCreator;
MNN_PUBLIC const RuntimeCreator *MNNGetExtraRuntimeCreator(MNNForwardType type);
} // namespace MNN

struct MnnRuntimeEngine {
    std::unique_ptr<MNN::Interpreter> interpreter;
    MNN::Session *session = nullptr;
    std::vector<std::pair<std::string, MNN::Tensor *>> inputs;
    std::vector<std::pair<std::string, MNN::Tensor *>> outputs;
    std::string last_error;
    std::vector<std::unique_ptr<MNN::Tensor>> input_hosts;
    std::vector<std::unique_ptr<MNN::Tensor>> output_hosts;
    int effective_threads = 0;

    ~MnnRuntimeEngine() {
        if (session != nullptr && interpreter != nullptr) {
            interpreter->releaseSession(session);
        }
    }
};

namespace {
thread_local std::string creation_error;

void set_error(MnnRuntimeEngine *engine, std::string message) {
    if (engine != nullptr) {
        engine->last_error = std::move(message);
    } else {
        creation_error = std::move(message);
    }
}

size_t logical_elements(const MNN::Tensor *tensor) {
    size_t count = 1;
    for (const int dimension : tensor->shape()) {
        if (dimension <= 0 || count > SIZE_MAX / static_cast<size_t>(dimension)) {
            return 0;
        }
        count *= static_cast<size_t>(dimension);
    }
    return count;
}

bool is_f32(const MNN::Tensor *tensor) {
    const auto type = tensor->getType();
    return type.code == halide_type_float && type.bits == 32 && type.lanes == 1;
}

std::vector<std::pair<std::string, MNN::Tensor *>> tensor_list(
    const std::map<std::string, MNN::Tensor *> &tensors) {
    std::vector<std::pair<std::string, MNN::Tensor *>> result;
    result.reserve(tensors.size());
    for (const auto &tensor : tensors) {
        result.emplace_back(tensor.first, tensor.second);
    }
    return result;
}

const std::vector<std::pair<std::string, MNN::Tensor *>> *select_tensors(
    const MnnRuntimeEngine *engine,
    int32_t input) {
    if (engine == nullptr) {
        return nullptr;
    }
    return input != 0 ? &engine->inputs : &engine->outputs;
}

void configure_backend(MNN::BackendConfig &backend, const MnnRuntimeConfig &config) {
    backend.precision = static_cast<MNN::BackendConfig::PrecisionMode>(config.precision);
    backend.power = static_cast<MNN::BackendConfig::PowerMode>(config.power);
    backend.memory = static_cast<MNN::BackendConfig::MemoryMode>(config.memory);
}
} // namespace

extern "C" {

const char *mnn_runtime_version(void) { return MNN::getVersion(); }

int32_t mnn_runtime_backend_available(int32_t backend) {
    if (backend == MNN_FORWARD_AUTO) {
        return 1;
    }
    if (backend < MNN_FORWARD_CPU || backend >= MNN_FORWARD_ALL) {
        return 0;
    }
    return MNN::MNNGetExtraRuntimeCreator(static_cast<MNNForwardType>(backend)) != nullptr;
}

MnnRuntimeEngine *mnn_runtime_engine_create(
    const void *model,
    size_t model_size,
    const MnnRuntimeConfig *config) {
    creation_error.clear();
    if (model == nullptr || model_size == 0 || model_size > static_cast<size_t>(INT32_MAX) || config == nullptr || config->threads <= 0) {
        set_error(nullptr, "model bytes and a positive thread count are required");
        return nullptr;
    }

    if (config->precision < 0 || config->precision > 3 || config->power < 0 || config->power > 2 || config->memory < 0 || config->memory > 2) {
        set_error(nullptr, "invalid backend configuration enum");
        return nullptr;
    }
    try {
        auto engine = std::make_unique<MnnRuntimeEngine>();
        engine->interpreter.reset(MNN::Interpreter::createFromBuffer(model, model_size));
        if (engine->interpreter == nullptr) {
            set_error(nullptr, "MNN could not parse the model buffer");
            return nullptr;
        }
        engine->interpreter->setSessionMode(MNN::Interpreter::Session_Release);

        MNN::BackendConfig backend_config;
        configure_backend(backend_config, *config);
        MNN::ScheduleConfig schedule;
        schedule.type = static_cast<MNNForwardType>(config->backend);
        schedule.backupType = MNN_FORWARD_CPU;
        if (schedule.type == MNN_FORWARD_OPENCL || schedule.type == MNN_FORWARD_VULKAN) {
            schedule.mode = config->gpu_mode;
        } else {
            schedule.numThread = config->threads;
        }
        schedule.backendConfig = &backend_config;

        engine->session = engine->interpreter->createSession(schedule);
        if (engine->session == nullptr) {
            set_error(nullptr, "MNN could not create an inference session");
            return nullptr;
        }

        int resize_status = 0;
        if (!engine->interpreter->getSessionInfo(engine->session, MNN::Interpreter::RESIZE_STATUS, &resize_status) || resize_status != 0) {
            set_error(nullptr, "MNN session could not allocate/resize its tensors");
            return nullptr;
        }
        int backends[2] = {-1, -1};
        engine->interpreter->getSessionInfo(engine->session, MNN::Interpreter::BACKENDS, backends);
        if (backends[0] == MNN_FORWARD_CPU) {
            engine->interpreter->getSessionInfo(engine->session, MNN::Interpreter::THREAD_NUMBER, &engine->effective_threads);
        }
        engine->inputs = tensor_list(engine->interpreter->getSessionInputAll(engine->session));
        engine->outputs = tensor_list(engine->interpreter->getSessionOutputAll(engine->session));
        if (engine->inputs.empty() || engine->outputs.empty()) {
            set_error(nullptr, "model must expose at least one input and one output");
            return nullptr;
        }
        for (const auto &tensor : engine->inputs) {
            if (!is_f32(tensor.second)) {
                set_error(nullptr, "this release supports only f32 input tensors");
                return nullptr;
            }
        }
        for (const auto &tensor : engine->outputs) {
            if (!is_f32(tensor.second)) {
                set_error(nullptr, "this release supports only f32 output tensors");
                return nullptr;
            }
        }
        for (const auto &entry : engine->inputs) {
            if (logical_elements(entry.second) == 0 || logical_elements(entry.second) > INT32_MAX / sizeof(float)) {
                set_error(nullptr, "input shape is dynamic or exceeds the native allocation limit");
                return nullptr;
            }
            engine->input_hosts.emplace_back(new MNN::Tensor(entry.second, entry.second->getDimensionType()));
        }
        for (const auto &entry : engine->outputs) {
            if (logical_elements(entry.second) == 0 || logical_elements(entry.second) > INT32_MAX / sizeof(float)) {
                set_error(nullptr, "output shape is dynamic or exceeds the native allocation limit");
                return nullptr;
            }
            engine->output_hosts.emplace_back(new MNN::Tensor(entry.second, entry.second->getDimensionType()));
        }
        // Static sessions own their weights/executions after successful creation.
        engine->interpreter->releaseModel();
        return engine.release();
    } catch (const std::exception &error) {
        set_error(nullptr, error.what());
    } catch (...) {
        set_error(nullptr, "unknown exception while creating MNN engine");
    }
    return nullptr;
}

int32_t mnn_runtime_effective_threads(const MnnRuntimeEngine *engine) {
    return engine == nullptr ? 0 : engine->effective_threads;
}

int32_t mnn_runtime_tensor_layout(const MnnRuntimeEngine *engine, int32_t input, size_t index) {
    const auto *tensors = select_tensors(engine, input);
    if (tensors == nullptr || index >= tensors->size()) return -1;
    return (*tensors)[index].second->getDimensionType() == MNN::Tensor::TENSORFLOW ? 1 : 0;
}

void mnn_runtime_engine_destroy(MnnRuntimeEngine *engine) { delete engine; }

const char *mnn_runtime_last_error(const MnnRuntimeEngine *engine) {
    return engine == nullptr ? creation_error.c_str() : engine->last_error.c_str();
}

size_t mnn_runtime_tensor_count(const MnnRuntimeEngine *engine, int32_t input) {
    const auto *tensors = select_tensors(engine, input);
    return tensors == nullptr ? 0 : tensors->size();
}

const char *mnn_runtime_tensor_name(
    const MnnRuntimeEngine *engine,
    int32_t input,
    size_t index) {
    const auto *tensors = select_tensors(engine, input);
    if (tensors == nullptr || index >= tensors->size()) {
        return nullptr;
    }
    return (*tensors)[index].first.c_str();
}

size_t mnn_runtime_tensor_rank(
    const MnnRuntimeEngine *engine,
    int32_t input,
    size_t index) {
    const auto *tensors = select_tensors(engine, input);
    if (tensors == nullptr || index >= tensors->size()) {
        return 0;
    }
    return (*tensors)[index].second->shape().size();
}

int32_t mnn_runtime_tensor_shape(
    const MnnRuntimeEngine *engine,
    int32_t input,
    size_t index,
    int32_t *dimensions,
    size_t capacity) {
    const auto *tensors = select_tensors(engine, input);
    if (tensors == nullptr || index >= tensors->size() || dimensions == nullptr) {
        return MNN_RUNTIME_INVALID_ARGUMENT;
    }
    const auto shape = (*tensors)[index].second->shape();
    if (capacity < shape.size()) {
        return MNN_RUNTIME_SHAPE_ERROR;
    }
    std::copy(shape.begin(), shape.end(), dimensions);
    return MNN_RUNTIME_OK;
}

int32_t mnn_runtime_write_input_index_f32(
    MnnRuntimeEngine *engine,
    size_t index,
    const float *data,
    size_t element_count) {
    if (engine == nullptr || data == nullptr) {
        return MNN_RUNTIME_INVALID_ARGUMENT;
    }
    try {
        MNN::Tensor *device = index < engine->inputs.size() ? engine->inputs[index].second : nullptr;
        if (device == nullptr) {
            set_error(engine, "input tensor not found");
            return MNN_RUNTIME_TENSOR_NOT_FOUND;
        }
        if (!is_f32(device)) {
            set_error(engine, "input tensor is not f32");
            return MNN_RUNTIME_TENSOR_TYPE_ERROR;
        }
        const size_t expected = logical_elements(device);
        if (expected == 0 || expected != element_count) {
            set_error(engine, "input tensor element count does not match model shape");
            return MNN_RUNTIME_SHAPE_ERROR;
        }

        const auto &host = engine->input_hosts[index];
        if (host == nullptr || host->host<float>() == nullptr) {
            set_error(engine, "could not allocate host input tensor");
            return MNN_RUNTIME_COPY_ERROR;
        }
        std::memcpy(host->host<float>(), data, element_count * sizeof(float));
        if (!device->copyFromHostTensor(host.get())) {
            set_error(engine, "MNN failed to copy the host input tensor");
            return MNN_RUNTIME_COPY_ERROR;
        }
        return MNN_RUNTIME_OK;
    } catch (const std::exception &error) {
        set_error(engine, error.what());
    } catch (...) {
        set_error(engine, "unknown exception while copying input tensor");
    }
    return MNN_RUNTIME_INTERNAL_ERROR;
}

int32_t mnn_runtime_run(MnnRuntimeEngine *engine) {
    if (engine == nullptr) {
        return MNN_RUNTIME_INVALID_ARGUMENT;
    }
    try {
        const auto status = engine->interpreter->runSession(engine->session);
        if (status != MNN::NO_ERROR) {
            set_error(
                engine,
                "MNN runSession returned error code " + std::to_string(static_cast<int>(status)));
            return MNN_RUNTIME_INFERENCE_ERROR;
        }
        return MNN_RUNTIME_OK;
    } catch (const std::exception &error) {
        set_error(engine, error.what());
    } catch (...) {
        set_error(engine, "unknown exception while running inference");
    }
    return MNN_RUNTIME_INTERNAL_ERROR;
}

int32_t mnn_runtime_read_output_index_f32(
    MnnRuntimeEngine *engine,
    size_t index,
    float *data,
    size_t element_count) {
    if (engine == nullptr || data == nullptr) {
        return MNN_RUNTIME_INVALID_ARGUMENT;
    }
    try {
        MNN::Tensor *device = index < engine->outputs.size() ? engine->outputs[index].second : nullptr;
        if (device == nullptr) {
            set_error(engine, "output tensor not found");
            return MNN_RUNTIME_TENSOR_NOT_FOUND;
        }
        if (!is_f32(device)) {
            set_error(engine, "output tensor is not f32");
            return MNN_RUNTIME_TENSOR_TYPE_ERROR;
        }
        const size_t expected = logical_elements(device);
        if (expected == 0 || expected != element_count) {
            set_error(engine, "output buffer element count does not match model shape");
            return MNN_RUNTIME_SHAPE_ERROR;
        }

        const auto &host = engine->output_hosts[index];
        if (host == nullptr || host->host<float>() == nullptr) {
            set_error(engine, "MNN failed to copy the output tensor to host memory");
            return MNN_RUNTIME_COPY_ERROR;
        }
        if (!device->copyToHostTensor(host.get())) {
            set_error(engine, "MNN failed to copy the output tensor to host memory");
            return MNN_RUNTIME_COPY_ERROR;
        }
        std::memcpy(data, host->host<float>(), element_count * sizeof(float));
        return MNN_RUNTIME_OK;
    } catch (const std::exception &error) {
        set_error(engine, error.what());
    } catch (...) {
        set_error(engine, "unknown exception while copying output tensor");
    }
    return MNN_RUNTIME_INTERNAL_ERROR;
}

} // extern "C"
