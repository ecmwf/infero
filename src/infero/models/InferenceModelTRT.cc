/*
 * (C) Copyright 1996- ECMWF.
 *
 * This software is licensed under the terms of the Apache Licence Version 2.0
 * which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
 * In applying this licence, ECMWF does not waive the privileges and immunities
 * granted to it by virtue of its status as an intergovernmental organisation
 * nor does it submit to any jurisdiction.
 */

#include <functional>
#include <iostream>
#include <numeric>

#include "eckit/exception/Exceptions.h"
#include "eckit/log/Log.h"

#include "infero/models/InferenceModelTRT.h"


using namespace eckit;

namespace infero {

static InferenceModelBuilder<InferenceModelTRT> TRTBuilder;


InferenceModelTRT::InferenceModelTRT(const eckit::Configuration& conf) :
    InferenceModel(conf), Engine_(nullptr), Network_(nullptr) {

    // Model configuration
    readConfig_(conf);

    std::string ModelPath(ModelConfig_->getString("path"));

    // read/bcast model by mpi (when possible)
    broadcast_model(ModelPath);

    InferRuntime_ = nvinfer1::createInferRuntime(sample::gLogger.getTRTLogger());

    // if not null, use the model buffer
    if (modelBuffer_.size()){
        Log::info() << "Constructing ONNX model from buffer.." << std::endl;
        Log::info() << "Model expected size: " + std::to_string(modelBuffer_.size()) << std::endl;

        Engine_.reset(InferRuntime_->deserializeCudaEngine(modelBuffer_.data(), modelBuffer_.size()));

        if (!Engine_) {
            std::string err = "failed to read the TRT engine!";
            throw eckit::FailedSystemCall(err, Here());
        }

    } else {  // otherwise construct from model path

        // Resurrect the TRF model..
        std::stringstream gieModelStream;
        ifstream en(ModelPath.c_str());
        gieModelStream << en.rdbuf();
        en.close();

        Log::info() << "Reading TRT model from " << ModelPath.c_str() << std::endl;

        // support for stringstream deserialization was deprecated in TensorRT v2
        // instead, read the stringstream into a memory buffer and pass that to TRT.
        gieModelStream.seekg(0, std::ios::end);
        const long int modelSize = gieModelStream.tellg();
        gieModelStream.seekg(0, std::ios::beg);

        modelMem_ = (char*)malloc(modelSize);
        if (!modelMem_) {
            std::string err = "failed to allocate " + std::to_string(modelSize) + " bytes";
            throw eckit::FailedSystemCall(err, Here());
        }

        gieModelStream.read((char*)modelMem_, modelSize);

        Engine_.reset(InferRuntime_->deserializeCudaEngine((void*)modelMem_, modelSize, NULL));
        if (!Engine_) {
            std::string err = "failed to read the TRT engine!";
            throw eckit::FailedSystemCall(err, Here());
        }

        Log::info() << "modelSize " << modelSize << std::endl;

    }
}

InferenceModelTRT::~InferenceModelTRT() {

    if (modelMem_)
        delete modelMem_;
}

std::string InferenceModelTRT::name() const
{
    return std::string(this->type());
}


void InferenceModelTRT::infer_impl(eckit::linalg::TensorFloat& tIn, eckit::linalg::TensorFloat& tOut,
                                   std::string input_name, std::string output_name){


    Log::info() << "TRT inference " << std::endl;

    // =================== prediction ======================
    // Create RAII buffer manager object
    samplesCommon::BufferManager buffers(Engine_);
    auto context = SampleUniquePtr<nvinfer1::IExecutionContext>(Engine_->createExecutionContext());

    std::string input_tensor_name  = Engine_->getBindingName(0);
    std::string output_tensor_name = Engine_->getBindingName(1);

    float* hostDataBuffer = static_cast<float*>(buffers.getHostBuffer(input_tensor_name));
    ::memcpy(hostDataBuffer, tIn.data(), sizeof(float) * tIn.size());
    // ======================================================

    //  Memcpy host 2 device
    buffers.copyInputToDevice();

    // inference
    Log::info() << "executing inference.." << std::endl;
    bool status = context->executeV2(buffers.getDeviceBindings().data());
    if (!status) {
        throw eckit::SeriousBug("inference FAILED!", Here());
    }

    // Memcpy device 2 host
    buffers.copyOutputToHost();
    // ======================================================

    // ======================= output =======================    
    Log::info() << "Copying output...";

    eckit::Timing t_start(statistics_.timer_);
    float* output = static_cast<float*>(buffers.getHostBuffer(output_tensor_name));    
    if (tOut.isRight()) {
        // TRT uses Left (C) tensor layouts, so we need to convert
        eckit::linalg::TensorFloat tLeft(output, tOut.shape(), false);  // wrap data
        eckit::linalg::TensorFloat tRight = tLeft.transformLeftToRightLayout();
        tOut = tRight;
    }
    else {
        // TRT uses Left (C) tensor layouts, so we can copy straight into memory of tOut
        ::memcpy(tOut.data(), output, tOut.size() * sizeof(float));
    }
    statistics_.oTensorLayoutTiming_ += eckit::Timing{statistics_.timer_} - t_start;
    // ======================================================
}

void InferenceModelTRT::print(ostream &os) const
{
    os << "A TRT Model" << std::endl;
}


void infero::InferenceModelTRT::infer_mimo_impl(std::vector<eckit::linalg::TensorFloat*> &tIn, std::vector<const char*> &input_names,
                                                std::vector<eckit::linalg::TensorFloat*> &tOut, std::vector<const char*> &output_names)
{

    Log::info() << "TRT inference " << std::endl;

    samplesCommon::BufferManager buffers(Engine_);
    auto context = SampleUniquePtr<nvinfer1::IExecutionContext>(Engine_->createExecutionContext());

    // ====================== Input tensors ======================
    size_t NInputs = input_names.size();
    for (size_t i=0; i<NInputs; i++){

        // copy input data into buffer
        float* hostDataBuffer = static_cast<float*>(buffers.getHostBuffer(input_names[i]));
        ::memcpy(hostDataBuffer, tIn[i]->data(), sizeof(float) * tIn[i]->size());
    }
    // ===========================================================

    // ======================== inference ========================
    // Memcpy host 2 device
    buffers.copyInputToDevice();

    Log::info() << "executing inference.." << std::endl;
    bool status = context->executeV2(buffers.getDeviceBindings().data());
    if (!status) {
        throw eckit::SeriousBug("inference FAILED!", Here());
    }

    // Memcpy device 2 host
    buffers.copyOutputToHost();
    // ===========================================================

    // ====================== Output tensors ======================
    // N Output tensors
    size_t NOutputs = output_names.size();
    eckit::Timing t_start(statistics_.timer_);
    for (size_t i=0; i<NOutputs; i++){

        // output buffer
        float* output = static_cast<float*>(buffers.getHostBuffer(output_names[i]));

        if (tOut[i]->isRight()) {

            // TFC uses Left (C) tensor layouts, so we need to convert
            eckit::linalg::TensorFloat tLeft(output, tOut[i]->shape(), false);  // wrap data
            eckit::linalg::TensorFloat tRight = tLeft.transformLeftToRightLayout();
            *tOut[i] = tRight;

        } else {

            // TFC uses Left (C) tensor layouts, so we can copy straight into memory of tOut
            Log::info() << "output size " << tOut[i]->size() << std::endl;
            ::memcpy(tOut[i]->data(), output, tOut[i]->size() * sizeof(float));
        }
    }
    statistics_.oTensorLayoutTiming_ += eckit::Timing{statistics_.timer_} - t_start;
}


}  // namespace infero
