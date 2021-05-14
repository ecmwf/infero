/*
 * (C) Copyright 1996- ECMWF.
 * 
 * This software is licensed under the terms of the Apache Licence Version 2.0
 * which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
 * In applying this licence, ECMWF does not waive the privileges and immunities
 * granted to it by virtue of its status as an intergovernmental organisation
 * nor does it submit to any jurisdiction.
 */

#pragma once

#include <string>

#include "onnxruntime_cxx_api.h"

#include "infero/ml_engines/MLEngine.h"



class MLEngineONNX: public MLEngine
{    

public:    

    MLEngineONNX(std::string model_filename);

    ~MLEngineONNX();

    // run the inference
    std::unique_ptr<Tensor> infer(std::unique_ptr<Tensor>& input_sample);


private:


    std::unique_ptr<Ort::Session> session;
    Ort::SessionOptions session_options;

    // allocator
    Ort::AllocatorWithDefaultOptions allocator;

    // input layer info

    // note: we assume only one input node
    int input_node_idx;
    size_t num_input_nodes;
    char* input_name;
    std::vector<int64_t> input_node_dims;
    std::vector<const char*> input_node_names;

    // same as input size, but with "1" as batch dim
    // this makes sure that it can be used to allocate
    // the input tensor by the library
    std::vector<int64_t> input_node_dims_1;
    size_t input_shape_flat;

    // output layer info
    int output_node_idx;
    size_t num_output_nodes;
    char* output_name;
    std::vector<const char*> output_node_names;
    std::vector<int64_t> output_node_dims;
    std::vector<int64_t> output_node_dims_1;
    int64_t output_shape_flat;

    std::vector<float> data_buffer;

private:

    void input_setup(Ort::Session& session);

    void output_setup(Ort::Session& session);

    void print(std::ostream& os) const;

};
