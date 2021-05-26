/*
 * (C) Copyright 1996- ECMWF.
 *
 * This software is licensed under the terms of the Apache Licence Version 2.0
 * which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
 * In applying this licence, ECMWF does not waive the privileges and immunities
 * granted to it by virtue of its status as an intergovernmental organisation
 * nor does it submit to any jurisdiction.
 */

#include <chrono>
#include <cstdio>
#include <iostream>

#include "eckit/log/Log.h"

#include "infero/MLTensor.h"
#include "infero/ml_engines/MLEngineTFlite.h"


#define DATA_SCALAR_TYPE float
#define OUTPUT_SCALAR_TYPE float


#define TFLITE_MINIMAL_CHECK(x)                                  \
    if (!(x)) {                                                  \
        fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
        exit(1);                                                 \
    }

using namespace eckit;

namespace infero {


MLEngineTFlite::MLEngineTFlite(std::string model_filename) : MLEngine(model_filename) {
    // Load model
    model = tflite::FlatBufferModel::BuildFromFile(this->mModelFilename.c_str());
    TFLITE_MINIMAL_CHECK(model != nullptr);

    // Build the interpreter with the InterpreterBuilder.
    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder builder(*model, resolver);
    interpreter = std::unique_ptr<tflite::Interpreter>(new tflite::Interpreter);

    builder(&interpreter);
    TFLITE_MINIMAL_CHECK(interpreter != nullptr);

    // Allocate tensor buffers.
    TFLITE_MINIMAL_CHECK(interpreter->AllocateTensors() == kTfLiteOk);
    tflite::PrintInterpreterState(interpreter.get());
}

MLEngineTFlite::~MLEngineTFlite() {}

std::unique_ptr<infero::MLTensor> MLEngineTFlite::infer(std::unique_ptr<infero::MLTensor>& input_sample) {

    Log::info() << "Sample tensor shape: ";
    for (auto i : input_sample->shape())
        Log::info() << i << ", ";
    Log::info() << std::endl;

    // reshape the internal input tensor to accept the user passed input
    std::vector<int> sh_(input_sample->shape().size());
    for (int i = 0; i < input_sample->shape().size(); i++)
        sh_[i] = input_sample->shape()[i];

    interpreter->ResizeInputTensor(interpreter->inputs()[0], sh_);
    interpreter->AllocateTensors();

    // =========================== copy tensor ============================
    float* input      = interpreter->typed_input_tensor<float>(0);
    const float* data = input_sample->data();
    size_t data_size  = input_sample->size();
    for (size_t i = 0; i < data_size; i++) {
        *(input + i) = *(data + i);
    }
    // ====================================================================

    // ========================== Run inference ===========================
    TFLITE_MINIMAL_CHECK(interpreter->Invoke() == kTfLiteOk);

    printf("\n\n=== Post-invoke Interpreter State ===\n");
    tflite::PrintInterpreterState(interpreter.get());
    // ====================================================================

    // ========================== Get output ==============================
    float* output     = interpreter->typed_output_tensor<float>(0);
    TfLiteTensor* out = interpreter->output_tensor(0);

    std::vector<size_t> out_shape(out->dims->size);
    for (int i = 0; i < out->dims->size; i++)
        out_shape[i] = out->dims->data[i];

    Log::info() << "Output tensor shape: ";
    for (auto i : out_shape)
        Log::info() << i << ", ";
    Log::info() << std::endl;

    auto pred_ptr = std::unique_ptr<infero::MLTensor>(new infero::MLTensor(out_shape));

    // copy output data
    for (size_t i = 0; i < pred_ptr->size(); i++) {
        *(pred_ptr->data() + i) = *(output + i);
    }
    // ====================================================================

    return pred_ptr;
}

}  // namespace infero
