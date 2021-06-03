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

#include "infero/ml_engines/MLEngineTRT.h"


using namespace eckit;

namespace infero {


MLEngineTRT::MLEngineTRT(std::string model_filename) : MLEngine(model_filename), mEngine(nullptr), network(nullptr) {

    // Resurrect the TRF model..
    std::stringstream gieModelStream;
    ifstream en(mModelFilename.c_str());
    gieModelStream << en.rdbuf();
    en.close();

    Log::info() << "Reading TRT model from " << mModelFilename.c_str() << std::endl;

    // support for stringstream deserialization was deprecated in TensorRT v2
    // instead, read the stringstream into a memory buffer and pass that to TRT.
    gieModelStream.seekg(0, std::ios::end);
    const int modelSize = gieModelStream.tellg();
    gieModelStream.seekg(0, std::ios::beg);

    void* modelMem = malloc(modelSize);
    if (!modelMem) {
        std::string err = "failed to allocate " + std::to_string(modelSize) + " bytes";
        throw eckit::FailedSystemCall(err, Here());
    }

    gieModelStream.read((char*)modelMem, modelSize);
    infer_runtime = nvinfer1::createInferRuntime(sample::gLogger.getTRTLogger());
    mEngine.reset(infer_runtime->deserializeCudaEngine(modelMem, modelSize, NULL));
    if (!mEngine) {
        std::string err = "failed to read the TRT engine!";
        throw eckit::FailedSystemCall(err, Here());
    }

    Log::info() << "modelSize " << modelSize << std::endl;
}

MLEngineTRT::~MLEngineTRT() {}


std::unique_ptr<infero::MLTensor> MLEngineTRT::infer(std::unique_ptr<infero::MLTensor>& input_sample) {

    // =================== prediction ======================
    // Create RAII buffer manager object
    samplesCommon::BufferManager buffers(mEngine);
    auto context = SampleUniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());

    Log::info() << "mEngine->getNbBindings() " << mEngine->getNbBindings() << std::endl;
    Log::info() << "mEngine->getBindingName(0) " << mEngine->getBindingName(0) << std::endl;
    Log::info() << "mEngine->bindingIsInput(0) " << mEngine->bindingIsInput(0) << std::endl;
    Log::info() << "mEngine->getBindingDimensions(0) " << mEngine->getBindingDimensions(0) << std::endl;
    Log::info() << "mEngine->getBindingName(1) " << mEngine->getBindingName(1) << std::endl;
    Log::info() << "mEngine->getBindingDimensions(1) " << mEngine->getBindingDimensions(1) << std::endl;

    std::string input_tensor_name  = mEngine->getBindingName(0);
    std::string output_tensor_name = mEngine->getBindingName(1);
    //    Dims input_dims = mEngine->getBindingDimensions(0);
    Dims output_dims = mEngine->getBindingDimensions(1);

    //    auto output_tensor_name = network->getOutput(0)->getName();
    //    Log::info() << "output_tensor_name " << output_tensor_name << std::endl;

    float* hostDataBuffer = static_cast<float*>(buffers.getHostBuffer(input_tensor_name));
    float* data           = input_sample->data();
    size_t data_size      = input_sample->size();
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
    int shape_flat = 1;
    std::vector<size_t> shape(output_dims.nbDims);
    for (int i = 0; i < output_dims.nbDims; i++) {
        shape[i] = output_dims.d[i];
        shape_flat *= output_dims.d[i];
        Log::info() << "output_dims.d[i] " << output_dims.d[i] << std::endl;
    }

    // copy output data
    float* output = static_cast<float*>(buffers.getHostBuffer(output_tensor_name));
    auto pred_ptr = std::unique_ptr<infero::MLTensor>(new infero::MLTensor(shape, false));
    for (int i = 0; i < shape_flat; i++) {
        *(pred_ptr->data() + i) = *(output + i);
    }

    return pred_ptr;
    // ======================================================
}

}  // namespace infero
