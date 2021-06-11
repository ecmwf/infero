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

#include "infero/models/InferenceModel.h"


namespace infero {

class InferenceModelONNX : public InferenceModel {

public:

    InferenceModelONNX(const eckit::Configuration& conf);

    ~InferenceModelONNX();

protected:

    void infer(eckit::linalg::TensorFloat& tIn, eckit::linalg::TensorFloat& tOut);

    void print(std::ostream& os) const;

private:

    // ORT session
    std::unique_ptr<Ort::Session> session;
    std::unique_ptr<Ort::SessionOptions> session_options;
    std::unique_ptr<Ort::Env> env;

    // allocator
    Ort::AllocatorWithDefaultOptions allocator;

    // NOTE: we assume only one input node
    size_t input_node_idx;
    size_t num_input_nodes;
    char* input_name;
    std::vector<int64_t> input_layer_shape;
    std::vector<const char*> input_node_names;

    // NOTE: we assume only one output node
    size_t output_node_idx;
    size_t num_output_nodes;
    char* output_name;
    std::vector<int64_t> output_layer_shape;
    std::vector<const char*> output_node_names;

    // output data
    std::vector<float> data_buffer;

private:
    void queryInputLayer();

    void queryOutputLayer();

};

}  // namespace infero
