/*
 * (C) Copyright 1996- ECMWF.
 * 
 * This software is licensed under the terms of the Apache Licence Version 2.0
 * which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
 * In applying this licence, ECMWF does not waive the privileges and immunities
 * granted to it by virtue of its status as an intergovernmental organisation
 * nor does it submit to any jurisdiction.
 */

#include "infero/ml_engines/MLEngineONNX.h"

#include "eckit/log/Log.h"
#include <assert.h>
#include <iostream>
#include <chrono>

//using namespace std;
//using namespace std::chrono;
using namespace eckit;


MLEngineONNX::MLEngineONNX(std::string model_filename):
    MLEngine(model_filename)
{

}

MLEngineONNX::~MLEngineONNX()
{

}

int MLEngineONNX::build()
{
    return 0;
}

PredictionPtr MLEngineONNX::infer(InputDataPtr& input_sample)
{

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "onnx_model");
    Ort::SessionOptions session_options;

    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    Log::debug() << "mModelFilename " << mModelFilename << std::endl;
    Ort::Session session(env, mModelFilename.c_str(), session_options);

    // fill in input/output info
    this->_input_info(session);
    this->_output_info(session);


    input_shape_flat = 1;
    for(const auto& i: input_sample->shape)
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
                                                              input_sample->data(),
                                                              input_shape_flat,
                                                              input_node_dims_1.data(),
                                                              input_node_dims_1.size());

    assert(input_tensor.IsTensor());

    Log::debug() << "tensor created " << std::endl;

    // score model & input tensor, get back output tensor
//    auto start = high_resolution_clock::now();

//    for (int i=0; i<input_node_names.size(); i++)
//        Log::info() << "input_node_names[i] " << input_node_names[i] << std::endl;

    auto output_tensors = session.Run(Ort::RunOptions{nullptr},
                                      input_node_names.data(),
                                      &input_tensor,
                                      1,
                                      output_node_names.data(),
                                      1);        

//    auto stop = high_resolution_clock::now();
//    auto duration_inference = duration_cast<milliseconds>(stop - start);

    assert(output_tensors.size() == 1 && output_tensors.front().IsTensor());

    Log::debug() << "assertion passed! " << std::endl;
    Log::debug() << "output_shape_flat " << output_shape_flat << std::endl;

    // Get pointer to output tensor ML_SCALAR values
    float* floatarr = output_tensors.front().GetTensorMutableData<float>();

    Log::debug() << "assertion passed! " << std::endl;
    Log::debug() << "output_shape_flat " << output_shape_flat << std::endl;

//    for (int i=0; i<input_node_dims_1.size(); i++)
//        Log::info() << "input_node_dims_1[i] " << input_node_dims_1[i] << std::endl;

    std::vector<float> output_data(output_shape_flat);
//    std::vector<long unsigned int> shape(output_node_dims.size());

    for (size_t i=0; i<output_shape_flat; i++){
        output_data[i] = *(floatarr+i);
    }

    return PredictionPtr(new Prediction(output_data, output_node_dims_1));

}


void MLEngineONNX::_input_info(Ort::Session& session)
{

    num_input_nodes = session.GetInputCount();
    input_node_names.resize(num_input_nodes);

    printf("Number of inputs tensors = %zu\n", num_input_nodes);

    // iterate over all input nodes
    for (int i = 0; i < num_input_nodes; i++) {

        // print input node names
        char* input_name = session.GetInputName(i, allocator);
        printf("Input %d : name=%s\n", i, input_name);
        input_node_names[i] = input_name;

        // print input node types
        Ort::TypeInfo type_info = session.GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

        ONNXTensorElementDataType type = tensor_info.GetElementType();
        printf("Input %d : type=%d\n", i, type);

        // print input shapes/dims
        input_node_dims = tensor_info.GetShape();
        printf("Input %d : num_dims=%zu\n", i, input_node_dims.size());
        for (int j = 0; j < input_node_dims.size(); j++)
            printf("Input %d : dim %d=%jd\n", i, j, input_node_dims[j]);
    }

    input_node_dims_1.resize(input_node_dims.size());
    for (int i=0; i<input_node_dims.size(); i++)
        input_node_dims_1[i] = input_node_dims[i];
    input_node_dims_1[0] = 1;
}

void MLEngineONNX::_output_info(Ort::Session& session)
{
    // print number of model output nodes
    num_output_nodes = session.GetOutputCount();
    output_node_names.resize(num_output_nodes);

    printf("Number of output tensors = %zu\n", num_output_nodes);

    // iterate over all output nodes
    for (int i = 0; i < num_output_nodes; i++) {

        // print output node names
        char* output_name = session.GetOutputName(i, allocator);
        printf("Output %d : name=%s\n", i, output_name );

        output_node_names[i] = output_name;

        // print output node types
        Ort::TypeInfo type_info = session.GetOutputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

        ONNXTensorElementDataType type = tensor_info.GetElementType();
        printf("Output %d : type=%d\n", i, type);

        // print output shapes/dims
        output_node_dims = tensor_info.GetShape();
        printf("Output %d : num_dims=%zu\n", i, output_node_dims.size());
        for (int j = 0; j < output_node_dims.size(); j++)
            printf("Output %d : dim %d=%jd\n", i, j, output_node_dims[j]);
    }

    output_node_dims_1.resize(output_node_dims.size());
    for (int i=0; i<output_node_dims.size(); i++)
        output_node_dims_1[i] = output_node_dims[i];
    output_node_dims_1[0] = 1;
}
