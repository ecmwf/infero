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

#include <memory>
#include <ostream>
#include <string>
#include <fstream>

#include "eckit/config/Configuration.h"
#include "eckit/linalg/Tensor.h"
#include "eckit/log/Log.h"
#include "eckit/io/SharedBuffer.h"

#include "ModelStatistics.h"

using eckit::Log;

namespace infero {

/// Minimal interface for a inference model
class InferenceModel {


public:
    static InferenceModel* create(const std::string& type,
                                  const eckit::Configuration& conf);

    InferenceModel(const eckit::Configuration& conf);

    virtual ~InferenceModel();

    /// opens the engine
    virtual void open();

    /// run the inference
    virtual void infer(eckit::linalg::TensorFloat& tIn, eckit::linalg::TensorFloat& tOut,
                       std::string input_name = "", std::string output_name = "");

    virtual void infer_impl(eckit::linalg::TensorFloat& tIn, eckit::linalg::TensorFloat& tOut,
                            std::string input_name = "", std::string output_name = "");

    /// run the inference (multi-input/multi-output inference)
    virtual void infer_mimo(std::vector<eckit::linalg::TensorFloat*> &tIn, std::vector<const char*> &input_names,
                            std::vector<eckit::linalg::TensorFloat*> &tOut, std::vector<const char*> &output_names);

    virtual void infer_mimo_impl(std::vector<eckit::linalg::TensorFloat*> &tIn, std::vector<const char*> &input_names,
                                 std::vector<eckit::linalg::TensorFloat*> &tOut, std::vector<const char*> &output_names);

    /// closes the engine
    virtual void close();

    ModelStatistics& statistics(){ return statistics_; }

protected:
    /// print the model
    virtual void print(std::ostream& os) const = 0;

    friend std::ostream& operator<<(std::ostream& os, InferenceModel& obj) {
        obj.print(os);
        return os;
    }

    virtual void broadcast_model(const std::string path);

    eckit::SharedBuffer modelBuffer_;

    ModelStatistics statistics_;

private:

    bool isOpen_;
};

}  // namespace infero
