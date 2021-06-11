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


InferenceModelTRT::InferenceModelTRT(const eckit::Configuration& conf) :
    InferenceModel(), Engine_(nullptr), Network_(nullptr) {

    // Resurrect the TRF model..
    std::string ModelPath(conf.getString("path"));
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

    modelMem = (char*)malloc(modelSize);
    if (!modelMem) {
        std::string err = "failed to allocate " + std::to_string(modelSize) + " bytes";
        throw eckit::FailedSystemCall(err, Here());
    }

    gieModelStream.read((char*)modelMem, modelSize);
    InferRuntime_ = nvinfer1::createInferRuntime(sample::gLogger.getTRTLogger());
    Engine_.reset(InferRuntime_->deserializeCudaEngine((void*)modelMem, modelSize, NULL));
    if (!Engine_) {
        std::string err = "failed to read the TRT engine!";
        throw eckit::FailedSystemCall(err, Here());
    }

    Log::info() << "modelSize " << modelSize << std::endl;
}

InferenceModelTRT::~InferenceModelTRT() {

    if (modelMem)
        delete modelMem;
}


void InferenceModelTRT::infer(eckit::linalg::TensorFloat& tIn, eckit::linalg::TensorFloat& tOut){


    if (tIn.isRight()) {
        Log::info() << "Input Tensor has right-layout, but left-layout is needed. "
                    << "Transforming to left.." << std::endl;
        ;
        tIn.toLeftLayout();
    }
    Log::info() << "TRT inference " << std::endl;

    // =================== prediction ======================
    // Create RAII buffer manager object
    samplesCommon::BufferManager buffers(Engine_);
    auto context = SampleUniquePtr<nvinfer1::IExecutionContext>(Engine_->createExecutionContext());

    Log::info() << "mEngine->getNbBindings() " << Engine_->getNbBindings() << std::endl;
    Log::info() << "mEngine->getBindingName(0) " << Engine_->getBindingName(0) << std::endl;
    Log::info() << "mEngine->bindingIsInput(0) " << Engine_->bindingIsInput(0) << std::endl;
    Log::info() << "mEngine->getBindingDimensions(0) " << Engine_->getBindingDimensions(0) << std::endl;
    Log::info() << "mEngine->getBindingName(1) " << Engine_->getBindingName(1) << std::endl;
    Log::info() << "mEngine->getBindingDimensions(1) " << Engine_->getBindingDimensions(1) << std::endl;

    std::string input_tensor_name  = Engine_->getBindingName(0);
    std::string output_tensor_name = Engine_->getBindingName(1);
    //    Dims input_dims = mEngine->getBindingDimensions(0);
    Dims output_dims = Engine_->getBindingDimensions(1);

    //    auto output_tensor_name = network->getOutput(0)->getName();
    //    Log::info() << "output_tensor_name " << output_tensor_name << std::endl;

    float* hostDataBuffer = static_cast<float*>(buffers.getHostBuffer(input_tensor_name));
    float* data           = tIn.data();
    size_t data_size      = tIn.size();
    for (size_t i = 0; i < data_size; i++) {
        hostDataBuffer[i] = *(data + i);
    }
    // ======================================================

    // Memcpy from host input buffers to device input buffers
    buffers.copyInputToDevice();

    Log::info() << "executing inference.." << std::endl;
    bool status = context->executeV2(buffers.getDeviceBindings().data());
    if (!status) {
        throw eckit::SeriousBug("inference FAILED!", Here());
    }

    // Memcpy from device output buffers to host output buffers
    buffers.copyOutputToHost();
    // ======================================================

    // ======================= output =======================
    size_t shape_flat = 1;
    std::vector<size_t> shape(output_dims.nbDims);
    for (int i = 0; i < output_dims.nbDims; i++) {
        shape[i] = output_dims.d[i];
        shape_flat *= output_dims.d[i];
        Log::info() << "output_dims.d[i] " << output_dims.d[i] << std::endl;
    }

    // copy output data
    float* output = static_cast<float*>(buffers.getHostBuffer(output_tensor_name));

    // copy output data
    Log::info() << "Copying output...";
    ASSERT(tOut.shape() == shape);
    if (tOut.isRight()) {
        // TRT uses Left (C) tensor layouts, so we need to convert
        eckit::linalg::TensorFloat tLeft(output, shape, false);  // wrap data
        tOut = tLeft.transformLeftToRightLayout();  // creates temporary tensor with data in left layout
    }
    else {
        // TRT uses Left (C) tensor layouts, so we can copy straight into memory of tOut
        memcpy(tOut.data(), output, shape_flat * sizeof(float));
    }
    // ======================================================
}

void InferenceModelTRT::print(ostream &os) const
{
    os << "A TRT Model" << std::endl;
}


}  // namespace infero
