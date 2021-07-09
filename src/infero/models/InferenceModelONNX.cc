/*
 * (C) Copyright 1996- ECMWF.
 *
 * This software is licensed under the terms of the Apache Licence Version 2.0
 * which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
 * In applying this licence, ECMWF does not waive the privileges and immunities
 * granted to it by virtue of its status as an intergovernmental organisation
 * nor does it submit to any jurisdiction.
 */

#include <assert.h>
#include <algorithm>
#include <chrono>
#include <iostream>

#include "eckit/exception/Exceptions.h"
#include "eckit/log/Log.h"

#include "infero/models/InferenceModelONNX.h"
#include "infero/infero_utils.h"


using namespace eckit;

namespace infero {


InferenceModelONNX::InferenceModelONNX(const eckit::Configuration& conf) :
    InferenceModel(conf),
    inputName_(nullptr),
    outputName_(nullptr) {

    std::string ModelPath(conf.getString("path"));

    // environment
    env = std::unique_ptr<Ort::Env>(new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "onnx_model"));

    // Session options
    session_options = std::unique_ptr<Ort::SessionOptions>(new Ort::SessionOptions);
    session_options->SetIntraOpNumThreads(1);
    session_options->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    // if not null, use the model buffer
    if (model_buffer){
        Log::info() << "Constructing ONNX model from buffer.." << std::endl;
        Log::info() << "Model expected size: " + std::to_string(model_buffer->size()) << std::endl;
        session = std::unique_ptr<Ort::Session>(new Ort::Session(*env,
                                                                 model_buffer->as_void_ptr(),
                                                                 model_buffer->size(),
                                                                 *session_options));
    } else {  // otherwise construct from model path
        session = std::unique_ptr<Ort::Session>(new Ort::Session(*env, ModelPath.c_str(), *session_options));
    }


    // query input/output layers
    queryInputLayer();
    queryOutputLayer();
}

InferenceModelONNX::~InferenceModelONNX() {

    if (inputName_) {
        delete inputName_;
    }

    if (outputName_) {
        delete outputName_;
    }
}

void InferenceModelONNX::infer(TensorFloat& tIn, TensorFloat& tOut) {

    if (tIn.isRight()) {
        Log::info() << "Input Tensor has right-layout, but left-layout is needed. "
                    << "Transforming to left.." << std::endl;
        ;
        tIn.toLeftLayout();
    }
    Log::info() << "ONNX inference " << std::endl;

    // make a copy of the input data
    dataBuffer_.resize(tIn.size());
    for (int i = 0; i < tIn.size(); i++) {
        dataBuffer_[i] = tIn.data()[i];
    }

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    auto shape_64 = utils::convert_shape<size_t, int64_t>(tIn.shape());
    Ort::Value input_tensor =
        Ort::Value::CreateTensor<float>(memory_info, dataBuffer_.data(), tIn.size(), shape_64.data(), shape_64.size());
    ASSERT(input_tensor.IsTensor());

    Ort::TensorTypeAndShapeInfo info = input_tensor.GetTensorTypeAndShapeInfo();
    Log::info() << "Sample tensor shape: ";
    for (auto i : info.GetShape())
        Log::info() << i << ", ";
    Log::info() << std::endl;

    auto output_tensors = session->Run(Ort::RunOptions{nullptr}, inputNodeNames_.data(), &input_tensor,
                                       numInputNodes_, outputNodeNames_.data(), numOutputNodes_);

    ASSERT(output_tensors.size() == 1 && output_tensors.front().IsTensor());


    auto out_tensor_info = output_tensors.front().GetTensorTypeAndShapeInfo();
    size_t out_size      = 1;
    for (auto i : out_tensor_info.GetShape()) {
        out_size *= i;
    }

    Log::info() << "Prediction tensor shape: ";
    for (auto i : out_tensor_info.GetShape())
        Log::info() << i << ", ";
    Log::info() << std::endl;

    auto shape = utils::convert_shape<int64_t, size_t>(out_tensor_info.GetShape());
    const float* floatarr = output_tensors.front().GetTensorMutableData<float>();

    ASSERT(tOut.shape() == shape);
    if (tOut.isRight()) {
        // ONNX uses Left (C) tensor layouts, so we need to convert
        TensorFloat tLeft(floatarr, shape, false);  // wrap data
        tOut = tLeft.transformLeftToRightLayout();  // creates temporary tensor with data in left layout
    }
    else {
        // ONNX uses Left (C) tensor layouts, so we can copy straight into memory of tOut
        memcpy(tOut.data(), floatarr, out_size * sizeof(float));
    }
}


void InferenceModelONNX::queryInputLayer() {
    numInputNodes_ = session->GetInputCount();

    // note: for now we use the assumption
    // that there is only one input tensor to the network
    ASSERT(numInputNodes_ == 1);
    inputNodeIdx_ = 0;

    // get input name
    inputName_ = session->GetInputName(inputNodeIdx_, allocator);
    inputNodeNames_.push_back(inputName_);

    Ort::TypeInfo type_info                               = session->GetInputTypeInfo(inputNodeIdx_);
    Ort::Unowned<Ort::TensorTypeAndShapeInfo> tensor_info = type_info.GetTensorTypeAndShapeInfo();

    inputLayerShape_ = tensor_info.GetShape();
}


void InferenceModelONNX::queryOutputLayer() {
    numOutputNodes_ = session->GetOutputCount();

    // note: for now we use the assumption
    // that there is only one input tensor to the network
    ASSERT(numOutputNodes_ == 1);
    outputNodeIdx_ = 0;

    // print output node names
    outputName_ = session->GetOutputName(outputNodeIdx_, allocator);
    outputNodeNames_.push_back(outputName_);

    // NOTE: this is the shape of the output tensor as described by the model
    // so it can be "dynamic" (with -1, meaning that accepts any tensor size on that axis)
    Ort::TypeInfo type_info                               = session->GetOutputTypeInfo(outputNodeIdx_);
    Ort::Unowned<Ort::TensorTypeAndShapeInfo> tensor_info = type_info.GetTensorTypeAndShapeInfo();

    // print output shapes/dims
    outputLayerShape_ = tensor_info.GetShape();
}


void InferenceModelONNX::print(std::ostream& os) const {

    os << "N input tensors: " << numInputNodes_ << std::endl;
    os << "Input layer " << inputNodeNames_[0] << " expects a Tensor with " << inputLayerShape_.size()
       << " dimensions" << std::endl;

    for (int j = 0; j < inputLayerShape_.size(); j++)
        os << "dim [" << j << "]: " << inputLayerShape_[j] << std::endl;

    os << "N output tensors: " << numOutputNodes_ << std::endl;
    os << "Output layer " << outputNodeNames_[0] << " expects a Tensor with " << outputLayerShape_.size()
       << " dimensions" << std::endl;

    for (int j = 0; j < outputLayerShape_.size(); j++)
        os << "dim [" << j << "]: " << outputLayerShape_[j] << std::endl;
}

}  // namespace infero
