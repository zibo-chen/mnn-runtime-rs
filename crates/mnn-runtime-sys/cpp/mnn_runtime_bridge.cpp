#include "mnn_runtime_bridge.h"

#include <MNN/Interpreter.hpp>
#include <MNN/MNNDefine.h>
#include <MNN/Tensor.hpp>

#include <algorithm>
#include <cstring>
#include <fstream>
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
    bool ready = false;
    bool has_output = false;
    std::vector<bool> input_written;
    std::string cache_file;

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

bool session_ready(MnnRuntimeEngine *engine) {
    int status = -1;
    return engine->interpreter->getSessionInfo(
        engine->session, MNN::Interpreter::RESIZE_STATUS, &status) && status == 0;
}

void refresh_tensors(MnnRuntimeEngine *engine) {
    engine->inputs = tensor_list(engine->interpreter->getSessionInputAll(engine->session));
    engine->outputs = tensor_list(engine->interpreter->getSessionOutputAll(engine->session));
    engine->input_hosts.resize(engine->inputs.size());
    engine->input_written.resize(engine->inputs.size(), false);
    engine->output_hosts.resize(engine->outputs.size());
}

// Tensor::size uses signed int arithmetic. Budget for packed channels as well
// as the logical f32 values before asking MNN to allocate any input.
bool valid_shape(const std::vector<int> &shape) {
    if (shape.size() > 8) return false;
    size_t count = 1;
    for (const int dim : shape) {
        if (dim <= 0 || static_cast<size_t>(dim) > INT32_MAX / sizeof(float) / 4 / count) return false;
        count *= static_cast<size_t>(dim);
    }
    return true;
}

bool prepare_host(std::unique_ptr<MNN::Tensor> &host, const MNN::Tensor *device) {
    if (!is_f32(device) || !valid_shape(device->shape())) return false;
    const auto layout = device->getDimensionType(); // packed devices expose CAFFE (NCHW)
    if (!host || host->shape() != device->shape() || host->getDimensionType() != layout) {
        host.reset(new MNN::Tensor(device, layout));
    }
    return host && host->host<float>() != nullptr;
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
        // Defer allocation until we can validate the graph inputs. This also
        // avoids treating unresolved OCR dimensions as a creation failure.
        engine->interpreter->setSessionMode(MNN::Interpreter::Session_Resize_Defer);
        if (config->cache_file != nullptr && config->cache_file[0] != '\0') {
            engine->cache_file = config->cache_file;
            std::ofstream probe(engine->cache_file, std::ios::binary | std::ios::app);
            if (!probe) {
                set_error(nullptr, "cannot open GPU cache file for writing");
                return nullptr;
            }
            probe.close();
            engine->interpreter->setCacheFile(engine->cache_file.c_str());
        }

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

        int backends[2] = {-1, -1};
        engine->interpreter->getSessionInfo(engine->session, MNN::Interpreter::BACKENDS, backends);
        if (backends[0] == MNN_FORWARD_CPU) {
            engine->interpreter->getSessionInfo(engine->session, MNN::Interpreter::THREAD_NUMBER, &engine->effective_threads);
        }
        refresh_tensors(engine.get());
        if (engine->inputs.empty() || engine->outputs.empty()) {
            set_error(nullptr, "model must expose at least one input and one output");
            return nullptr;
        }
        bool dynamic = false;
        for (const auto &entry : engine->inputs) {
            if (!is_f32(entry.second)) {
                set_error(nullptr, "only f32 input tensors are supported");
                return nullptr;
            }
            const auto shape = entry.second->shape();
            if (std::any_of(shape.begin(), shape.end(), [](int d) { return d <= 0; })) {
                dynamic = true;
            } else if (!valid_shape(shape)) {
                set_error(nullptr, "input shape exceeds the native allocation limit");
                return nullptr;
            }
        }
        if (!dynamic) {
            engine->interpreter->resizeSession(engine->session);
            engine->ready = session_ready(engine.get());
            if (!engine->ready) {
                set_error(nullptr, "MNN session could not allocate/resize its tensors");
                return nullptr;
            }
            refresh_tensors(engine.get());
            for (const auto &output : engine->outputs) {
                if (!is_f32(output.second)) {
                    set_error(nullptr, "only f32 output tensors are supported");
                    return nullptr;
                }
            }
        }
        // Keep native model bytes: MNN refuses resizeSession after releaseModel.
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
    if (input == 0 && !engine->ready) return 1;
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
    if (input == 0 && !engine->ready) {
        if (capacity < 1) return MNN_RUNTIME_SHAPE_ERROR;
        dimensions[0] = -1;
        return MNN_RUNTIME_OK;
    }
    const auto shape = (*tensors)[index].second->shape();
    if (capacity < shape.size()) {
        return MNN_RUNTIME_SHAPE_ERROR;
    }
    std::copy(shape.begin(), shape.end(), dimensions);
    return MNN_RUNTIME_OK;
}

int32_t mnn_runtime_resize_inputs(
    MnnRuntimeEngine *engine, const MnnRuntimeInputShape *shapes, size_t count) {
    if (engine == nullptr || shapes == nullptr) return MNN_RUNTIME_INVALID_ARGUMENT;
    try {
        if (count != engine->inputs.size()) {
            set_error(engine, "resize requires every input exactly once");
            return MNN_RUNTIME_SHAPE_ERROR;
        }
        std::vector<bool> seen(count, false);
        std::vector<std::vector<int>> dimensions(count);
        bool changed = !engine->ready;
        for (size_t i = 0; i < count; ++i) {
            const auto &shape = shapes[i];
            if (shape.index >= count || seen[shape.index] || shape.dimensions == nullptr ||
                shape.rank > 8) {
                set_error(engine, "invalid or duplicate input shape");
                return MNN_RUNTIME_SHAPE_ERROR;
            }
            seen[shape.index] = true;
            auto &dims = dimensions[shape.index];
            dims.assign(shape.dimensions, shape.dimensions + shape.rank);
            if (!valid_shape(dims) || dims.size() != engine->inputs[shape.index].second->shape().size()) {
                set_error(engine, "input dimensions must be positive, retain rank, and fit native allocation limits");
                return MNN_RUNTIME_SHAPE_ERROR;
            }
            changed = changed || dims != engine->inputs[shape.index].second->shape();
        }
        engine->has_output = false;
        std::fill(engine->input_written.begin(), engine->input_written.end(), false);
        if (!changed) return MNN_RUNTIME_OK;
        engine->ready = false;
        for (size_t i = 0; i < count; ++i) {
            engine->interpreter->resizeTensor(engine->inputs[i].second, dimensions[i]);
        }
        engine->interpreter->resizeSession(engine->session);
        refresh_tensors(engine);
        engine->ready = session_ready(engine);
        if (!engine->ready) {
            set_error(engine, "MNN session could not allocate/resize its tensors for the supplied inputs");
            return MNN_RUNTIME_SESSION_ERROR;
        }
        return MNN_RUNTIME_OK;
    } catch (const std::exception &error) {
        set_error(engine, error.what());
    } catch (...) {
        set_error(engine, "unknown exception while resizing inputs");
    }
    engine->ready = false;
    return MNN_RUNTIME_INTERNAL_ERROR;
}

int32_t mnn_runtime_save_cache(MnnRuntimeEngine *engine) {
    if (engine == nullptr) return MNN_RUNTIME_INVALID_ARGUMENT;
    if (engine->cache_file.empty()) return MNN_RUNTIME_OK;
    try {
        // MNN 3.6 does not propagate file-open failures from updateCacheFile.
        std::ofstream probe(engine->cache_file, std::ios::binary | std::ios::app);
        if (!probe) {
            set_error(engine, "cannot open GPU cache file for writing");
            return MNN_RUNTIME_COPY_ERROR;
        }
        probe.close();
        const auto status = engine->interpreter->updateCacheFile(engine->session);
        if (status != MNN::NO_ERROR) {
            set_error(engine, "MNN failed to update GPU cache: " + std::to_string(status));
            return MNN_RUNTIME_SESSION_ERROR;
        }
        return MNN_RUNTIME_OK;
    } catch (const std::exception &error) {
        set_error(engine, error.what());
    } catch (...) {
        set_error(engine, "unknown exception while saving cache");
    }
    return MNN_RUNTIME_INTERNAL_ERROR;
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
        if (!engine->ready) {
            set_error(engine, "resize inputs successfully before writing tensors");
            return MNN_RUNTIME_SESSION_ERROR;
        }
        engine->has_output = false;
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

        auto &host = engine->input_hosts[index];
        if (!prepare_host(host, device)) {
            set_error(engine, "could not allocate host input tensor");
            return MNN_RUNTIME_COPY_ERROR;
        }
        std::memcpy(host->host<float>(), data, element_count * sizeof(float));
        if (!device->copyFromHostTensor(host.get())) {
            set_error(engine, "MNN failed to copy the host input tensor");
            return MNN_RUNTIME_COPY_ERROR;
        }
        engine->input_written[index] = true;
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
        engine->has_output = false;
        if (!engine->ready) {
            set_error(engine, "resize inputs successfully before running inference");
            return MNN_RUNTIME_SESSION_ERROR;
        }
        if (std::find(engine->input_written.begin(), engine->input_written.end(), false) != engine->input_written.end()) {
            set_error(engine, "write every input after resizing before running inference");
            return MNN_RUNTIME_INVALID_ARGUMENT;
        }
        const auto status = engine->interpreter->runSession(engine->session);
        if (status != MNN::NO_ERROR) {
            set_error(
                engine,
                "MNN runSession returned error code " + std::to_string(static_cast<int>(status)));
            return MNN_RUNTIME_INFERENCE_ERROR;
        }
        refresh_tensors(engine);
        for (const auto &output : engine->outputs) {
            if (!is_f32(output.second)) {
                set_error(engine, "only f32 output tensors are supported");
                return MNN_RUNTIME_TENSOR_TYPE_ERROR;
            }
        }
        engine->has_output = true;
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
        if (!engine->has_output) {
            set_error(engine, "no output is available from a successful inference");
            return MNN_RUNTIME_INFERENCE_ERROR;
        }
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

        auto &host = engine->output_hosts[index];
        if (!prepare_host(host, device)) {
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
