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


namespace infero {

class MLEngineONNX: public MLEngine
{    

public:    

    MLEngineONNX(std::string model_filename);

    ~MLEngineONNX();

    // run the inference
    virtual std::unique_ptr<infero::MLTensor> infer(std::unique_ptr<infero::MLTensor>& input_sample);


private:

    // ORT session
    std::unique_ptr<Ort::Session> session;
    std::unique_ptr<Ort::SessionOptions> session_options;
    std::unique_ptr<Ort::Env> env;

    // allocator
    Ort::AllocatorWithDefaultOptions allocator;

    // NOTE: we assume only one input node
    int input_node_idx;
    size_t num_input_nodes;
    char* input_name;
    std::vector<int64_t> input_layer_shape;
    std::vector<const char*> input_node_names;

    // NOTE: we assume only one output node
    int output_node_idx;
    size_t num_output_nodes;
    char* output_name;
    std::vector<int64_t> output_layer_shape;
    std::vector<const char*> output_node_names;

    // output data
    std::vector<float> data_buffer;

private:

    void query_input_layer();

    void query_output_layer();

    void print(std::ostream& os) const;

};

} // namespace linalg
