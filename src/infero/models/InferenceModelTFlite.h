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

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"

#include "infero/models/InferenceModel.h"


namespace infero {

class InferenceModelTFlite : public InferenceModel {

public:
    InferenceModelTFlite(const eckit::Configuration& conf,
                         const InferenceModelBuffer* model_buffer = nullptr);

    virtual ~InferenceModelTFlite();

protected:
    void infer(eckit::linalg::TensorFloat& tIn, eckit::linalg::TensorFloat& tOut);

    void print(std::ostream& os) const;

private:
    // TFlite model and interpreter
    std::unique_ptr<tflite::FlatBufferModel> model_;
    std::unique_ptr<tflite::Interpreter> interpreter_;
};

}  // namespace infero
