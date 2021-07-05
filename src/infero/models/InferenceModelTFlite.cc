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

#include "infero/models/InferenceModelTFlite.h"
#include "infero/infero_utils.h"


#define DATA_SCALAR_TYPE float
#define OUTPUT_SCALAR_TYPE float


#define TFLITE_MINIMAL_CHECK(x)                                  \
    if (!(x)) {                                                  \
        fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
        exit(1);                                                 \
    }

using namespace eckit;

namespace infero {


InferenceModelTFlite::InferenceModelTFlite(const eckit::Configuration& conf) :
    InferenceModel(conf) {

    std::string ModelPath(conf.getString("path"));

    // if not null, use the model buffer
    if (model_buffer){

        Log::info() << "Constructing TFLITE model from buffer.." << std::endl;
        Log::info() << "Model expected size: " + std::to_string(model_buffer->size()) << std::endl;
        model_ = tflite::FlatBufferModel::BuildFromBuffer((char*)model_buffer->data(),
                                                          model_buffer->size());

    } else {  // otherwise construct from model path
        model_ = tflite::FlatBufferModel::BuildFromFile(ModelPath.c_str());
    }

    TFLITE_MINIMAL_CHECK(model_ != nullptr);

    // Build the interpreter with the InterpreterBuilder.
    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder builder(*model_, resolver);
    interpreter_ = std::unique_ptr<tflite::Interpreter>(new tflite::Interpreter);

    builder(&interpreter_);
    TFLITE_MINIMAL_CHECK(interpreter_ != nullptr);

    // Allocate tensor buffers.
    TFLITE_MINIMAL_CHECK(interpreter_->AllocateTensors() == kTfLiteOk);
    tflite::PrintInterpreterState(interpreter_.get());
}

InferenceModelTFlite::~InferenceModelTFlite() {}


void InferenceModelTFlite::infer(eckit::linalg::TensorFloat& tIn, eckit::linalg::TensorFloat& tOut) {

    if (tIn.isRight()) {
        Log::info() << "Input Tensor has right-layout, but left-layout is needed. "
                    << "Transforming to left.." << std::endl;
        ;
        tIn.toLeftLayout();
    }

    Log::info() << "TFlite inference " << std::endl;

    Log::info() << "Sample tensor shape: ";
    for (auto i : tIn.shape())
        Log::info() << i << ", ";
    Log::info() << std::endl;

    // reshape the internal input tensor to accept the user passed input
    std::vector<int> sh_(tIn.shape().size());
    for (int i = 0; i < tIn.shape().size(); i++)
        sh_[i] = tIn.shape()[i];

    interpreter_->ResizeInputTensor(interpreter_->inputs()[0], sh_);
    interpreter_->AllocateTensors();

    // =========================== copy tensor ============================
    float* input      = interpreter_->typed_input_tensor<float>(0);
    const float* data = tIn.data();
    size_t data_size  = tIn.size();
    for (size_t i = 0; i < data_size; i++) {
        *(input + i) = *(data + i);
    }
    // ====================================================================

    // ========================== Run inference ===========================
    TFLITE_MINIMAL_CHECK(interpreter_->Invoke() == kTfLiteOk);

    printf("\n\n=== Post-invoke Interpreter State ===\n");
    tflite::PrintInterpreterState(interpreter_.get());
    // ====================================================================

    // ========================== Get output ==============================
    float* output     = interpreter_->typed_output_tensor<float>(0);
    TfLiteTensor* out = interpreter_->output_tensor(0);

    std::vector<size_t> out_shape(out->dims->size);
    size_t out_size = 1;
    for (int i = 0; i < out->dims->size; i++) {
        out_shape[i] = out->dims->data[i];
        out_size *= out_shape[i];
    }

    Log::info() << "Output tensor shape: ";
    for (auto i : out_shape)
        Log::info() << i << ", ";
    Log::info() << std::endl;

    // copy output data
    Log::info() << "Copying output..." << std::endl;
    ASSERT(tOut.shape() == out_shape);
    if (tOut.isRight()) {
        // TFlite uses Left (C) tensor layouts, so we need to convert
        TensorFloat tLeft(output, out_shape, false);  // wrap data
        tOut = tLeft.transformLeftToRightLayout();    // creates temporary tensor with data in left layout
    }
    else {
        // TFlite uses Left (C) tensor layouts, so we can copy straight into memory of tOut
        memcpy(tOut.data(), output, out_size * sizeof(float));
    }

    // ====================================================================
}

void InferenceModelTFlite::print(std::ostream &os) const
{
    os << "A TFlite Model" << std::endl;
}


}  // namespace infero
