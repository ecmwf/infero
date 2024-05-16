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

#include <torch/torch.h>
#include <torch/script.h>

#include "infero/models/InferenceModel.h"


namespace infero {

class InferenceModelTorch : public InferenceModel {

public:

    /**
     * Constructor
     * @param conf Configuration
     */
    InferenceModelTorch(const eckit::Configuration& conf);

    /**
     * name of the model
     */
    virtual std::string name() const override;

    /**
     * Type of the model
     */
    constexpr static const char* type() { return "torch"; }

    /**
     * print the model
     */
    void print(std::ostream& os) const override;

private:

    /**
     * inference implementation
     */
    void infer_impl(eckit::linalg::TensorFloat& tIn, eckit::linalg::TensorFloat& tOut,
                    std::string input_name = "", std::string output_name = "") override;

    /**
     * Multi-Input, Multi-output (MIMO) inference implementation
     */
    void infer_mimo_impl(std::vector<eckit::linalg::TensorFloat*> &tIn, std::vector<const char*> &input_names,
                        std::vector<eckit::linalg::TensorFloat*> &tOut, std::vector<const char*> &output_names) override;

    /**
     * Default configuration for the model
     */
    static eckit::LocalConfiguration defaultConfig();

private:

    torch::jit::script::Module torch_module_;
    
};

} // namespace infero