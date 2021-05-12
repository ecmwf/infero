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

#include "infero/ml_engines/MLEngine.h"
#include "infero/input_types/InputData.h"

#include "onnxruntime_cxx_api.h"


class MLEngineONNX: public MLEngine
{    

public:    

    MLEngineONNX(std::string model_filename);

    ~MLEngineONNX();

    // build the engine
    virtual int build();

    // run the inference
    PredictionPtr infer(InputDataPtr& input_sample);


private:

    // allocator
    Ort::AllocatorWithDefaultOptions allocator;

    // input layer info
    size_t num_input_nodes;
    std::vector<const char*> input_node_names;
    std::vector<int64_t> input_node_dims;

    // same as input size, but with "1" as batch dim
    // this makes sure that it can be used to allocate
    // the input tensor by the library
    std::vector<int64_t> input_node_dims_1;
    size_t input_shape_flat;

    // output layer info
    size_t num_output_nodes;
    std::vector<const char*> output_node_names;
    std::vector<int64_t> output_node_dims;
    std::vector<int64_t> output_node_dims_1;
    int64_t output_shape_flat;

private:

    void _input_info(Ort::Session& session);
    void _output_info(Ort::Session& session);
    void _save_prediction(std::vector<Ort::Value>& tensor,
                          std::string filename);

};
