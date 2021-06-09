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

#include "eckit/config/Configuration.h"
#include "eckit/linalg/Tensor.h"


namespace infero {

/// Minimal interface for a inference model
class InferenceModel {

public:

    static InferenceModel* create(const std::string& type, const eckit::Configuration& conf);

    InferenceModel();

    virtual ~InferenceModel();

    /// opens the engine
    virtual void open();

    /// run the inference
    virtual void infer(eckit::linalg::TensorFloat& tIn, eckit::linalg::TensorFloat& tOut) = 0;

    /// closes the engine
    virtual void close();

protected:
    /// print the model
    virtual void print(std::ostream& os) const = 0;

private:
    bool isOpen_;
};

friend std::ostream& operator<<(std::ostream& os, InferenceModel& obj) {
    obj.print(os);
    return os;
}

}  // namespace infero
