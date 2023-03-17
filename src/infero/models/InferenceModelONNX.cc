/*
 * (C) Copyright 1996- ECMWF.
 *
 * This software is licensed under the terms of the Apache Licence Version 2.0
 * which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
 * In applying this licence, ECMWF does not waive the privileges and immunities
 * granted to it by virtue of its status as an intergovernmental organisation
 * nor does it submit to any jurisdiction.
 */

#include <assert.h>
#include <algorithm>
#include <chrono>
#include <iostream>

#include "eckit/exception/Exceptions.h"
#include "eckit/log/Log.h"

#include "infero/models/InferenceModelONNX.h"
#include "infero/infero_utils.h"
#include "InferenceModelONNX.h"


using namespace eckit;

namespace infero {

static InferenceModelBuilder<InferenceModelONNX> onnxBuilder;


ModelParams_t InferenceModelONNX::implDefaultParams_(){
    ModelParams_t params_;
       
    params_["numInteropThreads"] = "1";
    params_["numIntraopThreads"] = "1";

    return params_;
}


InferenceModelONNX::InferenceModelONNX(const eckit::Configuration& conf) :
    InferenceModel(conf){

    // Model configuration
    readConfig_(conf);

    std::string ModelPath(ModelConfig_->getString("path"));

    // read/bcast model by mpi (when possible)
    broadcast_model(ModelPath);

    env = std::unique_ptr<Ort::Env>(new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "onnx_model"));

    // Session options
    session_options = std::unique_ptr<Ort::SessionOptions>(new Ort::SessionOptions);
    session_options->SetInterOpNumThreads(ModelConfig_->getInt("numInteropThreads"));
    session_options->SetIntraOpNumThreads(ModelConfig_->getInt("numIntraopThreads"));
    session_options->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    // if not null, use the model buffer
    if (modelBuffer_.size()){
        Log::info() << "Constructing ONNX model from buffer.." << std::endl;
        Log::info() << "Model expected size: " + std::to_string(modelBuffer_.size()) << std::endl;
        session = std::unique_ptr<Ort::Session>(new Ort::Session(*env,
                                                                 modelBuffer_.data(),
                                                                 modelBuffer_.size(),
                                                                 *session_options));
    } else {  // otherwise construct from model path
        session = std::unique_ptr<Ort::Session>(new Ort::Session(*env, ModelPath.c_str(), *session_options));
    }


    // setup input/output interface
    setupInputLayers();
    setupOutputLayers();
}

InferenceModelONNX::~InferenceModelONNX() {

    for (auto& n: inputNames){
        free (n);
    }

    for (auto& n: outputNames){
        free (n);
    }
}

std::string InferenceModelONNX::name() const
{
    return std::string(this->type());
}

void InferenceModelONNX::infer_impl(TensorFloat& tIn, TensorFloat& tOut,
                                    std::string input_name, std::string output_name) {

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    // only one input/output usable here
    ASSERT(inputNames.size() == 1);
    ASSERT(outputNames.size() == 1);

    auto shape_64 = utils::convert_shape<size_t, int64_t>(tIn.shape());
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info,
                                            tIn.data(),
                                            tIn.size(),
                                            shape_64.data(),
                                            shape_64.size());
    ASSERT(input_tensor.IsTensor());

    auto output_tensors = session->Run(Ort::RunOptions{nullptr},
                                       inputNames.data(),
                                       &input_tensor,
                                       numInputs,
                                       outputNames.data(),
                                       numOutputs);

    // output tensors
    ASSERT(output_tensors.size() == 1 && output_tensors.front().IsTensor());

    eckit::Timing t_start(statistics_.timer());
    if (tOut.layout() == eckit::linalg::TensorFloat::Layout::ColMajor) {

         // ONNX uses Left (C) tensor layouts, so we need to convert
         auto out_shape = output_tensors.front().GetTensorTypeAndShapeInfo().GetShape();
         TensorFloat tLeft(const_cast<float*>(output_tensors.front().GetTensorData<float>()),
                           utils::convert_shape<int64_t, size_t>(out_shape), 
                           eckit::linalg::TensorFloat::Layout::RowMajor);

         TensorFloat tRight = tLeft.transformRowMajorToColMajor();
         tOut = tRight;
    }

    else {
         // ONNX uses Left (C) tensor layouts, so we can copy straight into memory of tOut
         memcpy(tOut.data(), output_tensors.front().GetTensorData<float>(),
                output_tensors.front().GetTensorTypeAndShapeInfo().GetElementCount() * sizeof(float));
    }
    statistics_.oTensorLayoutTiming_ += eckit::Timing{statistics_.timer()} - t_start;

}

void InferenceModelONNX::infer_mimo_impl(std::vector<eckit::linalg::TensorFloat*> &tIn, std::vector<const char*> &input_names,
                                         std::vector<eckit::linalg::TensorFloat*> &tOut, std::vector<const char*> &output_names)
{

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<Ort::Value> inputTensors;

    // N Input tensors
    size_t NInputs = input_names.size();
    for (size_t i=0; i<NInputs; i++){

        auto shape_64 = utils::convert_shape<size_t, int64_t>(tIn[i]->shape());
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info,
                                                tIn[i]->data(),
                                                tIn[i]->size(),
                                                shape_64.data(),
                                                shape_64.size());
        ASSERT(input_tensor.IsTensor());

        inputTensors.emplace_back(std::move(input_tensor));

    }

    auto output_tensors = session->Run(Ort::RunOptions{nullptr},
                                       inputNames.data(),
                                       inputTensors.data(),
                                       numInputs,
                                       outputNames.data(),
                                       numOutputs);

    // output tensors
    ASSERT(output_tensors.size() == numOutputs);

    eckit::Timing t_start(statistics_.timer());
    for (size_t i=0; i<numOutputs; i++){

         ASSERT(output_tensors[i].IsTensor());

         if (tOut[i]->layout() == eckit::linalg::TensorFloat::Layout::ColMajor) {

             // ONNX uses Left (C) tensor layouts, so we need to convert
             auto out_shape = output_tensors[i].GetTensorTypeAndShapeInfo().GetShape();
             TensorFloat tLeft(const_cast<float*>(output_tensors[i].GetTensorData<float>()),
                               utils::convert_shape<int64_t, size_t>(out_shape),
                               eckit::linalg::TensorFloat::Layout::RowMajor);  // wrap data

             TensorFloat tRight = tLeft.transformRowMajorToColMajor();
             *tOut[i] = tRight;
         }
         else {
             // ONNX uses Left (C) tensor layouts, so we can copy straight into memory of tOut
             memcpy(tOut[i]->data(), output_tensors[i].GetTensorData<float>(),
                    output_tensors[i].GetTensorTypeAndShapeInfo().GetElementCount() * sizeof(float));
         }
    }
    statistics_.oTensorLayoutTiming_ += eckit::Timing{statistics_.timer()} - t_start;

}


void InferenceModelONNX::setupInputLayers() {

    // get input name
    numInputs = session->GetInputCount();

    for (size_t i=0; i<numInputs; i++){

        char* inputName_ = session->GetInputName(i, allocator);
        inputNames.push_back(inputName_);

        Ort::TypeInfo type_info = session->GetInputTypeInfo(i);
        Ort::Unowned<Ort::TensorTypeAndShapeInfo> tensor_info = type_info.GetTensorTypeAndShapeInfo();

        std::vector<int64_t> inputLayerShape_ = tensor_info.GetShape();
        inputLayerShapes.push_back(inputLayerShape_);
    }
}


void InferenceModelONNX::setupOutputLayers() {

    // get input name
    numOutputs = session->GetOutputCount();

    for (size_t i=0; i<numOutputs; i++){

        char* outputName_ = session->GetOutputName(i, allocator);
        outputNames.push_back(outputName_);

        Ort::TypeInfo type_info = session->GetOutputTypeInfo(i);
        Ort::Unowned<Ort::TensorTypeAndShapeInfo> tensor_info = type_info.GetTensorTypeAndShapeInfo();

        std::vector<int64_t> outputLayerShape_ = tensor_info.GetShape();
        outputLayerShapes.push_back(outputLayerShape_);

    }
}


void InferenceModelONNX::print(std::ostream& os) const {

    os << "ONNX model has: " << numInputs << " inputs" << std::endl;
    for (size_t i=0; i<numInputs; i++){
        Log::info() << "Layer [" << i << "] " << inputNames[i] << " has shape: ";
        for (auto s: inputLayerShapes[i]){
            Log::info() << s << ", ";
        }
        Log::info() << std::endl;
    }

    os << "ONNX model has: " << numOutputs << " outputs" << std::endl;
    for (size_t i=0; i<numOutputs; i++){
        Log::info() << "Layer [" << i << "] " << outputNames[i] << " has shape: ";
        for (auto s: outputLayerShapes[i]){
            Log::info() << s << ", ";
        }
        Log::info() << std::endl;
    }

}


void InferenceModelONNX::print_shape(const Ort::Value& t){
    Ort::TensorTypeAndShapeInfo info = t.GetTensorTypeAndShapeInfo();
    for (auto i : info.GetShape())
        Log::info() << i << ", ";
    Log::info() << std::endl;
}

}  // namespace infero
