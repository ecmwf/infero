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
#include <iostream>
#include <chrono>
#include <algorithm>

#include "eckit/log/Log.h"
#include "eckit/exception/Exceptions.h"

#include "infero/ml_engines/MLEngineONNX.h"

using namespace eckit;


MLEngineONNX::MLEngineONNX(std::string model_filename):
    MLEngine(model_filename)
{

    // Session options
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "onnx_model");
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    Log::debug() << "mModelFilename " << mModelFilename << std::endl;
    session = std::unique_ptr<Ort::Session>( new Ort::Session(env, mModelFilename.c_str(), session_options));

    // query input/output layers
    query_input_layer();
    query_output_layer();
}

MLEngineONNX::~MLEngineONNX()
{

}

std::unique_ptr<Tensor> MLEngineONNX::infer(std::unique_ptr<Tensor>& input_sample)
{

    // make a copy of the input data
    data_buffer.resize(input_sample->size());
    for (int i=0; i<input_sample->size(); i++){
        data_buffer[i] = input_sample->data()[i];
    }

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info,
                                                              data_buffer.data(),
                                                              input_sample->size(),
                                                              input_sample->shape().data(),
                                                              input_sample->shape().size());

    ASSERT(input_tensor.IsTensor());

    auto output_tensors = session->Run(Ort::RunOptions{nullptr},
                                       input_node_names.data(),
                                       &input_tensor,
                                       num_input_nodes,
                                       output_node_names.data(),
                                       num_output_nodes);

    ASSERT(output_tensors.size() == 1 && output_tensors.front().IsTensor());

    auto out_tensor_info = output_tensors.front().GetTensorTypeAndShapeInfo();
    int out_size = 1;
    for(auto i: out_tensor_info.GetShape())
        out_size *= i;

    // copy output data
    float* floatarr = output_tensors.front().GetTensorMutableData<float>();
    std::vector<float> output_data(out_size);
    for (size_t i=0; i<out_size; i++){
        output_data[i] = *(floatarr+i);
    }

    // return a Tensor
    return std::unique_ptr<Tensor>(new Tensor(output_data, out_tensor_info.GetShape()));
}


void MLEngineONNX::query_input_layer()
{
    num_input_nodes = session->GetInputCount();

    // note: for now we use the assumption
    // that there is only one input tensor to the network
    ASSERT(num_input_nodes == 1);
    input_node_idx = 0;

    // get input name
    input_name = session->GetInputName(input_node_idx, allocator);
    input_node_names.push_back(input_name);

    Ort::TypeInfo type_info = session->GetInputTypeInfo(input_node_idx);
    Ort::Unowned<Ort::TensorTypeAndShapeInfo> tensor_info = type_info.GetTensorTypeAndShapeInfo();

    input_layer_shape = tensor_info.GetShape();
}


void MLEngineONNX::query_output_layer()
{
    num_output_nodes = session->GetOutputCount();

    // note: for now we use the assumption
    // that there is only one input tensor to the network
    ASSERT(num_output_nodes == 1);
    output_node_idx = 0;

    // print output node names
    output_name = session->GetOutputName(output_node_idx, allocator);
    output_node_names.push_back(output_name);

    // NOTE: this is the shape of the output tensor as described by the model
    // so it can be "dynamic" (with -1, meaning that accepts any tensor size on that axis)
    Ort::TypeInfo type_info = session->GetOutputTypeInfo(output_node_idx);
    Ort::Unowned<Ort::TensorTypeAndShapeInfo> tensor_info = type_info.GetTensorTypeAndShapeInfo();

    // print output shapes/dims
    output_layer_shape = tensor_info.GetShape();
}


void MLEngineONNX::print(std::ostream& os) const {

    os << "N input tensors: " << num_input_nodes << std::endl;
    os << "Input layer expects a Tensor with "
       << input_layer_shape.size() << " dimensions" << std::endl;

    for (int j = 0; j < input_layer_shape.size(); j++)
        os << "dim [" << j << "]: " << input_layer_shape[j] << std::endl;

    os << "N output tensors: " << num_output_nodes << std::endl;
    os << "Input layer expects a Tensor with "
       << output_layer_shape.size() << " dimensions" << std::endl;

    for (int j = 0; j < output_layer_shape.size(); j++)
        os << "dim [" << j << "]: " << output_layer_shape[j] << std::endl;
}



