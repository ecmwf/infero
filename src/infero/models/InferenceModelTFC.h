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

    virtual ~InferenceModelTFC();

protected:
    void infer(eckit::linalg::TensorFloat& tIn, eckit::linalg::TensorFloat& tOut,
               std::string input_name = "serving_default_input_1", std::string output_name = "StatefulPartitionedCall");

    virtual void infer_mimo(std::vector<eckit::linalg::TensorFloat*> tIn, std::vector<const char*> input_names,
                            std::vector<eckit::linalg::TensorFloat*> tOut, std::vector<const char*> output_names);

    void print(std::ostream& os) const;

    virtual void broadcast_model(const std::string path);

private:

    void check_status(const TF_Status* s, std::string name);
    TF_Output getOperation(std::string name);
    TF_Tensor *TF_TensorFromData(const std::vector<size_t> &dims, float *data);

private:

    TF_Session* session;
    TF_Graph* network_graph;
    TF_Status* err_status;
    TF_SessionOptions* session_options;
    TF_Buffer* run_options;

};

}  // namespace infero
