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

#include "infero/inference_models/InferenceModel.h"


namespace infero {

class InferenceModelTFlite : public InferenceModel {

public:

    InferenceModelTFlite(std::string model_filename);

    virtual ~InferenceModelTFlite();

protected:

    // run the inference
    void do_infer(TensorFloat& tIn, TensorFloat& tOut);

private:

    void set_input_layout(TensorFloat& tIn);

    // TFlite model and interpreter
    std::unique_ptr<tflite::FlatBufferModel> model;
    std::unique_ptr<tflite::Interpreter> interpreter;
};

}  // namespace infero
