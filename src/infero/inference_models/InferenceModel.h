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


using namespace std;
using namespace eckit::linalg;


namespace infero {

/// Minimal interface for a inference model
class InferenceModel {

public:

    InferenceModel(std::string model_filename) : mModelFilename(model_filename) {}

    virtual ~InferenceModel();

    /// open handle
    static InferenceModel* open(std::string choice, const eckit::Configuration& conf);

    /// run the inference
    void infer(TensorFloat& tIn, TensorFloat& tOut);

    /// close the handle
    static void close(InferenceModel* handle);

    friend std::ostream& operator<<(std::ostream& os, InferenceModel& obj) {
        obj.print(os);
        return os;
    }


protected:

    /// print the model
    virtual void print(std::ostream& os) const {}

    /// set the correct input tensor layout
    /// as requested by the specific model
    virtual void set_input_layout(TensorFloat& tIn) = 0;

    // do run inference
    virtual void do_infer(TensorFloat& tIn, TensorFloat& tOut) = 0;

protected:

    std::string mModelFilename;

};


}  // namespace infero
