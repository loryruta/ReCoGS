#pragma once

#include <string>

#include <NvInfer.h>

BEGIN_NAMESPACE

class SimpleNvInferLogger : public nvinfer1::ILogger
{
public:
    const std::string name;

    explicit SimpleNvInferLogger(const std::string& name) noexcept : name(name) {}

    void log(Severity severity, const char* message) noexcept override
    {
        switch (severity) {
        case Severity::kINTERNAL_ERROR:
        case Severity::kERROR:
            printf("[ERROR] [%s] (NvInfer) %s\n", name.c_str(), message);
            break;
        case Severity::kWARNING:
            printf("[WARN ] [%s] (NvInfer) %s\n", name.c_str(), message);
            break;
        case Severity::kINFO:
            printf("[INFO ] [%s] (NvInfer) %s\n", name.c_str(), message);
            break;
        case Severity::kVERBOSE:
            printf("[DEBUG] [%s] (NvInfer) %s\n", name.c_str(), message);
            break;
        }
    }
};

inline std::string to_string(nvinfer1::Dims dims)
{
    std::string str = "(";
    for (int i = 0; i < dims.nbDims; ++i) {
        if (i > 0) str += ", ";
        str += std::to_string(dims.d[i]);
    }
    str += ")";
    return str;
}

inline std::string to_string(nvinfer1::DataType data_type)
{
    switch (data_type) {
    case nvinfer1::DataType::kFLOAT:
        return "FLOAT";
    case nvinfer1::DataType::kHALF:
        return "HALF";
    case nvinfer1::DataType::kINT8:
        return "INT8";
    case nvinfer1::DataType::kINT32:
        return "INT32";
    case nvinfer1::DataType::kBOOL:
        return "BOOL";
    case nvinfer1::DataType::kUINT8:
        return "UINT8";
    case nvinfer1::DataType::kFP8:
        return "FP8";
    case nvinfer1::DataType::kBF16:
        return "BF16";
    case nvinfer1::DataType::kINT64:
        return "INT64";
    case nvinfer1::DataType::kINT4:
        return "INT4";
    case nvinfer1::DataType::kFP4:
        return "FP4";
    }
}

inline std::string to_string(nvinfer1::TensorIOMode mode)
{
    switch (mode) {
    case nvinfer1::TensorIOMode::kNONE:
        return "NONE";
    case nvinfer1::TensorIOMode::kINPUT:
        return "INPUT";
    case nvinfer1::TensorIOMode::kOUTPUT:
        return "OUTPUT";
    }
}

inline std::string to_string(nvinfer1::TensorLocation location)
{
    switch (location) {
    case nvinfer1::TensorLocation::kHOST:
        return "HOST";
    case nvinfer1::TensorLocation::kDEVICE:
        return "DEVICE";
    }
}

END_NAMESPACE
