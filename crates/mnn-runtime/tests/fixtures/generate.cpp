#include "MNN_generated.h"
#include <memory>
#include <fstream>
#include <iostream>

std::vector<uint8_t> make_model(bool packed_input, int channels, bool nhwc = false) {
    MNN::NetT net;
    net.tensorName = {"input", "converted", "output"};
    net.tensorNumber = 3;
    net.outputName = {"output"};
    auto input = std::make_unique<MNN::OpT>();
    input->type = MNN::OpType_Input;
    input->name = "input";
    input->outputIndexes = {0};
    MNN::InputT input_info;
    input_info.dims = nhwc ? std::vector<int>{1, 1, 2, channels} : std::vector<int>{1, channels, 1, 2};
    input_info.dformat = nhwc ? MNN::MNN_DATA_FORMAT_NHWC : packed_input ? MNN::MNN_DATA_FORMAT_NC4HW4 : MNN::MNN_DATA_FORMAT_NCHW;
    input->main.Set(std::move(input_info));
    net.oplists.emplace_back(std::move(input));
    auto convert = std::make_unique<MNN::OpT>();
    convert->type = MNN::OpType_ConvertTensor;
    convert->name = "convert";
    convert->inputIndexes = {0};
    convert->outputIndexes = {1};
    MNN::TensorConvertInfoT convert_info;
    convert_info.source = nhwc ? MNN::MNN_DATA_FORMAT_NHWC : packed_input ? MNN::MNN_DATA_FORMAT_NC4HW4 : MNN::MNN_DATA_FORMAT_NCHW;
    convert_info.dest = (nhwc || packed_input) ? MNN::MNN_DATA_FORMAT_NCHW : MNN::MNN_DATA_FORMAT_NC4HW4;
    convert->main.Set(std::move(convert_info));
    net.oplists.emplace_back(std::move(convert));
    auto relu = std::make_unique<MNN::OpT>();
    relu->type = MNN::OpType_ReLU;
    relu->name = "relu";
    relu->inputIndexes = {1};
    relu->outputIndexes = {2};
    relu->main.Set(MNN::ReluT{});
    net.oplists.emplace_back(std::move(relu));
    flatbuffers::FlatBufferBuilder builder;
    builder.Finish(MNN::Net::Pack(builder, &net));
    return {builder.GetBufferPointer(), builder.GetBufferPointer() + builder.GetSize()};
}

int main(int argc, char **argv) {
    if (argc != 2) return 1;
    for (int channels : {3, 5}) for (int variant : {0, 1, 2}) {
        const auto bytes = make_model(variant == 1, channels, variant == 2);
        const auto path = std::string(argv[1]) + "/" + (variant == 0 ? "packed_output" : variant == 1 ? "packed_input" : "nhwc") + std::to_string(channels) + ".mnn";
        std::ofstream(path, std::ios::binary).write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    }
}
