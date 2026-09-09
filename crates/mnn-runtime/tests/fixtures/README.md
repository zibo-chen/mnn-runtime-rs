# Generated fixtures

These tiny, weight-free graphs are owned by this project. They contain Input,
ConvertTensor and ReLU operators, including packed tensors with 3 and 5 channels.
`generate.cpp` uses the MNN 3.6.0 generated schema and bundled FlatBuffers headers.

To regenerate, compile with C++17 and include `$MNN_SOURCE/schema/current` and
`$MNN_SOURCE/3rd_party/flatbuffers/include`, then pass this directory to the
resulting program. No MNN library or converter binary is required.
