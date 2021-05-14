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

#include "eckit/log/Log.h"
#include "eckit/exception/Exceptions.h"

#include "infero/ml_engines/MLEngineONNX.h"


//using namespace std;
//using namespace std::chrono;
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

    // fill in input/output layer
    input_setup(*session);
    output_setup(*session);
}

MLEngineONNX::~MLEngineONNX()
{

}

std::unique_ptr<Tensor> MLEngineONNX::infer(std::unique_ptr<Tensor>& input_sample)
{

    //---------------- input --------------------
    input_node_dims_1.resize(input_node_dims.size());
    for (int i=0; i<input_node_dims.size(); i++)
        input_node_dims_1[i] = input_node_dims[i];
    input_node_dims_1[0] = 1;

    // make a copy of the input data
    data_buffer.resize(input_sample->size());
    for (int i=0; i<input_sample->size(); i++){
        data_buffer[i] = input_sample->data()[i];
    }
    // ------------------------------------------

    //---------------- output -------------------
    output_node_dims_1.resize(output_node_dims.size());
    for (int i=0; i<output_node_dims.size(); i++)
        output_node_dims_1[i] = output_node_dims[i];
    output_node_dims_1[0] = 1;
    // ------------------------------------------

    input_shape_flat = 1;
    for(const auto& i: input_sample->shape())
        input_shape_flat *= i;

    Log::debug() << "input_shape_flat " << input_shape_flat << std::endl;

    output_shape_flat = 1;
    for(const auto& i: output_node_dims_1){
        Log::debug() << "output_node_dims_1 " << i << std::endl;
        output_shape_flat *= i;
    }
    Log::debug() << "output_shape_flat " << output_shape_flat << std::endl;

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info,
                                                              data_buffer.data(),
                                                              input_shape_flat,
                                                              input_node_dims_1.data(),
                                                              input_node_dims_1.size());

    assert(input_tensor.IsTensor());

    Log::debug() << "tensor created " << std::endl;

    auto output_tensors = session->Run(Ort::RunOptions{nullptr},
                                       input_node_names.data(),
                                       &input_tensor,
                                       1,
                                       output_node_names.data(),
                                       1);

    assert(output_tensors.size() == 1 && output_tensors.front().IsTensor());

    float* floatarr = output_tensors.front().GetTensorMutableData<float>();

    std::vector<float> output_data(output_shape_flat);

    for (size_t i=0; i<output_shape_flat; i++){
        output_data[i] = *(floatarr+i);
    }

    return std::unique_ptr<Tensor>(new Tensor(output_data, output_node_dims_1));

}


void MLEngineONNX::input_setup(Ort::Session& session)
{

    num_input_nodes = session.GetInputCount();

    // note: for now we use the assumption
    // that there is only one input tensor to the network
    ASSERT(num_input_nodes == 1);
    input_node_idx = 0;

    // get input name
    input_name = session.GetInputName(input_node_idx, allocator);
    input_node_names.push_back(input_name);

    Ort::TypeInfo type_info = session.GetInputTypeInfo(input_node_idx);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    ONNXTensorElementDataType type = tensor_info.GetElementType();

    input_node_dims = tensor_info.GetShape();

}

void MLEngineONNX::output_setup(Ort::Session& session)
{

    output_node_idx = 0;

    // print number of model output nodes
    num_output_nodes = session.GetOutputCount();

    // note: for now we use the assumption
    // that there is only one input tensor to the network
    ASSERT(num_output_nodes == 1);

    // print output node names
    output_name = session.GetOutputName(output_node_idx, allocator);
    output_node_names.push_back(output_name);

    // print output node types
    Ort::TypeInfo type_info = session.GetOutputTypeInfo(output_node_idx);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    ONNXTensorElementDataType type = tensor_info.GetElementType();

    // print output shapes/dims
    output_node_dims = tensor_info.GetShape();
}

void MLEngineONNX::print(std::ostream& os) const {

    os << "N input tensors: " << num_input_nodes << std::endl;
    os << "Input layer expects a Tensor with "
       << input_node_dims.size() << " dimensions" << std::endl;

    for (int j = 0; j < input_node_dims.size(); j++)
        os << "dim [" << j << "]: " << input_node_dims[j] << std::endl;

    os << "N output tensors: " << num_output_nodes << std::endl;
    os << "Input layer expects a Tensor with "
       << output_node_dims.size() << " dimensions" << std::endl;

    for (int j = 0; j < output_node_dims.size(); j++)
        os << "dim [" << j << "]: " << output_node_dims[j] << std::endl;
}

















