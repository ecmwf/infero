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


using namespace eckit;

namespace infero {

static InferenceModelBuilder<InferenceModelTFlite> tfliteBuilder;


InferenceModelTFlite::InferenceModelTFlite(const eckit::Configuration& conf) :
    InferenceModel(conf) {

    // Model configuration
    readConfig_(conf);

    std::string ModelPath(ModelConfig_->getString("path"));

    // read/bcast model by mpi (when possible)
    broadcast_model(ModelPath);

    // if not null, use the model buffer
    if (modelBuffer_.size()){

        Log::info() << "Constructing TFLITE model from buffer.." << std::endl;
        Log::info() << "Model expected size: " + std::to_string(modelBuffer_.size()) << std::endl;
        model_ = tflite::FlatBufferModel::BuildFromBuffer((char*)modelBuffer_.data(),
                                                          modelBuffer_.size());

    } else {  // otherwise construct from model path
        model_ = tflite::FlatBufferModel::BuildFromFile(ModelPath.c_str());
    }

    INFERO_CHECK(model_ != nullptr);

    // Build the interpreter with the InterpreterBuilder.
    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder builder(*model_, resolver);
    interpreter_ = std::unique_ptr<tflite::Interpreter>(new tflite::Interpreter);

    builder(&interpreter_);
    INFERO_CHECK(interpreter_ != nullptr);

    // Allocate tensor buffers.
    INFERO_CHECK(interpreter_->AllocateTensors() == kTfLiteOk);
    tflite::PrintInterpreterState(interpreter_.get());
}

InferenceModelTFlite::~InferenceModelTFlite() {}

std::string InferenceModelTFlite::name() const
{
    return std::string(this->type());
}


void InferenceModelTFlite::infer_impl(eckit::linalg::TensorFloat& tIn, eckit::linalg::TensorFloat& tOut,
                                      std::string input_name, std::string output_name) {

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
    ::memcpy(input, tIn.data(), sizeof(float) * tIn.size());
    // ====================================================================

    // ========================== Run inference ===========================
    INFERO_CHECK(interpreter_->Invoke() == kTfLiteOk);

//    printf("\n\n=== Post-invoke Interpreter State ===\n");
//    tflite::PrintInterpreterState(interpreter_.get());
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
    eckit::Timing t_start(statistics_.timer_);
    ASSERT(tOut.shape() == out_shape);
    if (tOut.isRight()) {
        // TFlite uses Left (C) tensor layouts, so we need to convert
        TensorFloat tLeft(output, out_shape, false);  // wrap data
        TensorFloat tRight = tLeft.transformLeftToRightLayout();
        tOut = tRight;
    }
    else {
        // TFlite uses Left (C) tensor layouts, so we can copy straight into memory of tOut
        memcpy(tOut.data(), output, out_size * sizeof(float));
    }
    statistics_.oTensorLayoutTiming_ += eckit::Timing{statistics_.timer_} - t_start;
    // ====================================================================
}


void InferenceModelTFlite::infer_mimo_impl(std::vector<eckit::linalg::TensorFloat*> &tIn, std::vector<const char*> &input_names,
                                           std::vector<eckit::linalg::TensorFloat*> &tOut, std::vector<const char*> &output_names)
{

    TfLiteStatus status_;

    // input tensors
    size_t NInputs = input_names.size();
    std::cout << "NInputs: " << NInputs << std::endl;
    for (size_t i=0; i<NInputs; i++){

        ASSERT(input_names[i] == std::string(interpreter_->input_tensor(i)->name));
        status_ = interpreter_->ResizeInputTensor(interpreter_->inputs()[i],
                                                  utils::convert_shape<size_t, int>(tIn[i]->shape()) );

        if(status_ != kTfLiteOk){
            throw Exception("Input Tensor " + std::string(interpreter_->input_tensor(i)->name)
                            + " failed to resize!");
        }
    }

    // Do the actual Input tensor allocation
    interpreter_->AllocateTensors();

    // Copy input data in the model buffer
    for (size_t i=0; i<NInputs; i++){
        float* input = interpreter_->typed_tensor<float>(i);
        ASSERT(input);
        ::memcpy(input, tIn[i]->data(), sizeof(float) * tIn[i]->size());
    }

    // Run inference
    INFERO_CHECK(interpreter_->Invoke() == kTfLiteOk);

    // copy output
    size_t NOutputs = output_names.size();
    eckit::Timing t_start(statistics_.timer_);
    for (size_t i=0; i<NOutputs; i++){

        std::cout << "Processing output: " << output_names[i] << std::endl;
        std::cout << "--> got output with name: " << interpreter_->output_tensor(i)->name << std::endl;

        float* output     = interpreter_->typed_output_tensor<float>(i);

        // copy output data
        Log::info() << "Copying output..." << std::endl;

        if (tOut[i]->isRight()) {
            // TFlite uses Left (C) tensor layouts, so we need to convert
            TensorFloat tLeft(output, tOut[i]->shape(), false);  // wrap data
            TensorFloat tRight = tLeft.transformLeftToRightLayout();
            *tOut[i] = tRight;
        }
        else {
            // TFlite uses Left (C) tensor layouts, so we can copy straight into memory of tOut
            memcpy(tOut[i]->data(), output, tOut[i]->size() * sizeof(float));
        }
    }

    statistics_.oTensorLayoutTiming_ += eckit::Timing{statistics_.timer_} - t_start;
}

void InferenceModelTFlite::print(std::ostream &os) const
{
    os << "A TFlite Model" << std::endl;
}


}  // namespace infero
