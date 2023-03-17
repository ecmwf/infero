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

    ~InferenceModelONNX() override;

    virtual std::string name() const override;

    constexpr static const char* type() { return "onnx"; }

    void print(std::ostream& os) const override;

    virtual ModelParams_t implDefaultParams_() override;

private:

    void infer_impl(eckit::linalg::TensorFloat& tIn, eckit::linalg::TensorFloat& tOut,
                    std::string input_name = "", std::string output_name = "") override;

    void infer_mimo_impl(std::vector<eckit::linalg::TensorFloat*> &tIn, std::vector<const char*> &input_names,
                         std::vector<eckit::linalg::TensorFloat*> &tOut, std::vector<const char*> &output_names) override;

private:

    // ORT session
    std::unique_ptr<Ort::Session> session;
    std::unique_ptr<Ort::SessionOptions> session_options;
    std::unique_ptr<Ort::Env> env;

    // allocator
    Ort::AllocatorWithDefaultOptions allocator;

    // input interface
    size_t numInputs;
    std::vector<char*> inputNames;    
    std::vector<std::vector<int64_t>> inputLayerShapes;

    // output interface
    size_t numOutputs;
    std::vector<char*> outputNames;
    std::vector<Ort::Value> outputTensors;
    std::vector<std::vector<int64_t>> outputLayerShapes;


private:
    void setupInputLayers();

    void setupOutputLayers();
    void print_shape(const Ort::Value& t);

};

}  // namespace infero
