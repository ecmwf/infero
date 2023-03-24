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

#include "tensorflow/c/c_api.h"

#include "infero/models/InferenceModel.h"


namespace infero {

class InferenceModelTFC : public InferenceModel {

public:
    InferenceModelTFC(const eckit::Configuration& conf);

    ~InferenceModelTFC() override;

    virtual std::string name() const override;

    constexpr static const char* type() { return "tf_c"; }

    void print(std::ostream& os) const override;

private:

    void infer_impl(eckit::linalg::TensorFloat& tIn, eckit::linalg::TensorFloat& tOut,
                    std::string input_name = "", std::string output_name = "") override;

    void infer_mimo_impl(std::vector<eckit::linalg::TensorFloat*> &tIn, std::vector<const char*> &input_names,
                         std::vector<eckit::linalg::TensorFloat*> &tOut, std::vector<const char*> &output_names) override;

    void check_status(const TF_Status* s, std::string name);    
    TF_Tensor *TF_TensorFromData(const std::vector<size_t> &dims, float *data);

    /// Get an oeration tensor buffer from layer name
    TF_Output GetOperationBuffer_(std::string name, int op_id = 0);

    /// Get an oeration tensor buffer from layer name
    /// + contains specialised logic for input layer
    TF_Output GetInputOperationBuffer_(std::string name);

    /// Get an oeration tensor buffer from layer name
    /// + contains specialised logic for output layer
    TF_Output GetOutputOperationBuffer_(std::string name);


    virtual void broadcast_model(const std::string path);

    static eckit::LocalConfiguration defaultConfig();

    // configure session options from model configuration
    void configureSessionOptions();

private:

    TF_Session* session;
    TF_Graph* network_graph;
    TF_Status* err_status;
    TF_SessionOptions* session_options;
    TF_Buffer* run_options;
};

}  // namespace infero
