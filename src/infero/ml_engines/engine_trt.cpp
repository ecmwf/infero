/*
 * (C) Copyright 1996- ECMWF.
 * 
 * This software is licensed under the terms of the Apache Licence Version 2.0
 * which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
 * In applying this licence, ECMWF does not waive the privileges and immunities
 * granted to it by virtue of its status as an intergovernmental organisation
 * nor does it submit to any jurisdiction.
 */

#include "infero/ml_engines/engine_trt.h"

#include "eckit/log/Log.h"

#include <iostream>


using namespace eckit;


MLEngineTRT::MLEngineTRT(std::string model_filename):    
    MLEngine(model_filename),
    mEngine(nullptr),
    network(nullptr){

    this->build();

}

MLEngineTRT::~MLEngineTRT()
{

}

int MLEngineTRT::build()
{

    // Resurrect the TRF model..
    std::stringstream gieModelStream;
    ifstream en(mModelFilename.c_str());
    gieModelStream << en.rdbuf();
    en.close();

    std::cout << "reading model from " << mModelFilename.c_str() << std::endl;

    // support for stringstream deserialization was deprecated in TensorRT v2
    // instead, read the stringstream into a memory buffer and pass that to TRT.
    gieModelStream.seekg(0, std::ios::end);
    const int modelSize = gieModelStream.tellg();
    gieModelStream.seekg(0, std::ios::beg);

    void* modelMem = malloc(modelSize);

    if( !modelMem )
    {
        std::cerr << "failed to allocate "
                  << modelSize
                  << " bytes to deserialize model" << std::endl;
        return -1;
    }

    gieModelStream.read((char*)modelMem, modelSize);
    nvinfer1::IRuntime* infer = nvinfer1::createInferRuntime(sample::gLogger.getTRTLogger());
    mEngine.reset(infer->deserializeCudaEngine(modelMem, modelSize, NULL));

    std::cout << "modelSize " << modelSize << std::endl;

    free(modelMem);

    return 0;
}


MLEngineTRTPtr MLEngineTRT::from_onnx(std::string onnx_path,
                                      TRTOptions& options,
                                      std::string trt_path){

    auto builder = SampleUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger()));
    if (!builder)
    {
        Log::error() << "builder FAILED! " << std::endl;
        return nullptr;
    }

    const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = SampleUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    if (!network)
    {
        Log::error() << "network FAILED! " << std::endl;
        return nullptr;
    }

    auto config = SampleUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config)
    {
        Log::error() << "config FAILED! " << std::endl;
        return nullptr;
    }

    auto parser = SampleUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, sample::gLogger.getTRTLogger()));
    if (!parser)
    {
        Log::error() << "parser FAILED! " << std::endl;
        return nullptr;
    }

    std::cout << "constructing network.." << std::endl;


    // =================== constructing the network.. =========================
    IOptimizationProfile* profile = builder.get()->createOptimizationProfile();

    // NB this is the optimization step, for now e just keep the input to be
    // made of one and one sample only..
    const char* input_layer = options.output_layer_name.c_str();
    profile->setDimensions(input_layer, OptProfileSelector::kMIN, Vector2Dims(options.InputDimsMin) );
    profile->setDimensions(input_layer, OptProfileSelector::kOPT, Vector2Dims(options.InputDimsOpt) );
    profile->setDimensions(input_layer, OptProfileSelector::kMAX, Vector2Dims(options.InputDimsMax) );
    config->addOptimizationProfile(profile);

    std::cout << "parsing file " << onnx_path << std::endl;
    auto parsed = parser->parseFromFile(onnx_path.c_str(),
                                        static_cast<int>(sample::gLogger.getReportableSeverity()));

    if (!parsed)
    {
        std::cout << "parsed  FAILED! " << std::endl;
        return nullptr;
    }

    config->setMaxWorkspaceSize(options.workspace_size);
    config->setFlag(BuilderFlag::kTF32);

    // build the network
    auto mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
                builder->buildEngineWithConfig(*network, *config), samplesCommon::InferDeleter());

    // save trt model
    auto trtModelStream = mEngine->serialize();
    ofstream p(trt_path);
    p.write((const char*)trtModelStream->data(),trtModelStream->size());
    p.close();

    return MLEngineTRTPtr(new MLEngineTRT("model.trt"));
}


// TODO still work in progress
PredictionPtr MLEngineTRT::infer(InputDataPtr& input_sample)
{

#if 0

    std::stringstream gieModelStream;
    ifstream en("model.trt");
    gieModelStream << en.rdbuf();
    en.close();

//    nvinfer1::ICudaEngine* mEngine = IRuntime::deserializeCudaEngine();
    nvinfer1::IRuntime* infer = nvinfer1::createInferRuntime(sample::gLogger.getTRTLogger());
//    mEngine = infer->deserializeCudaEngine(gieModelStream);
    // support for stringstream deserialization was deprecated in TensorRT v2
    // instead, read the stringstream into a memory buffer and pass that to TRT.
    gieModelStream.seekg(0, std::ios::end);
    const int modelSize = gieModelStream.tellg();
    gieModelStream.seekg(0, std::ios::beg);

    void* modelMem = malloc(modelSize);

    if( !modelMem )
    {
        std::cout << "failed to allocate "
                  << modelSize
                  << " bytes to deserialize model" << std::endl;
        return 0;
    }

    gieModelStream.read((char*)modelMem, modelSize);
    mEngine.reset(infer->deserializeCudaEngine(modelMem, modelSize, NULL));
    free(modelMem);


    // =================== prediction ======================
    // Create RAII buffer manager object
    samplesCommon::BufferManager buffers(mEngine);

    SampleUniquePtr<nvinfer1::IExecutionContext> context = SampleUniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());

//    Log::debug() << "context completed!  " << mParams.inputTensorNames.size() << std::endl;

    // Read the input data into the managed buffers
//    ASSERT(mParams.inputTensorNames.size() == 1);

    std::cout << "calling processInput.." << std::endl;

//    auto output_tensor_name = network->getOutput(0)->getName();
//    std::cout << "output_tensor_name " << output_tensor_name << std::endl;

    // TODO: how do we know output shape after loading up engine from model.trt??
    float* hostDataBuffer = static_cast<float*>(buffers.getHostBuffer("conv2d_17"));
    float* data = input_sample->get_data();
    size_t data_size = input_sample->get_size();
    std::cout << "Input data_size " << data_size << std::endl;
    for (size_t i = 0; i<data_size; i++){
        hostDataBuffer[i] = *(data+i);
    }
    std::cout << "Input copied to buffer!" << std::endl;
    // ======================================================

    // Memcpy from host input buffers to device input buffers
    std::cout << "copying data to device.." << std::endl;
    buffers.copyInputToDevice();

    std::cout << "executing inference.." << std::endl;
    bool status = context->executeV2(buffers.getDeviceBindings().data());
    if (!status)
    {
        Log::error() << "inference FAILED! " << std::endl;
    }

    // Memcpy from device output buffers to host output buffers
    std::cout << "copying data FROM device.." << std::endl;
    buffers.copyOutputToHost();
    // ======================================================

    // ==================== copy output =====================
    int shape_flat = 1;
    Dims out_dims = network->getOutput(0)->getDimensions();
    std::vector<int64_t> shape(out_dims.nbDims);
    for(int i=0; i<out_dims.nbDims; i++) {
        shape[i] = out_dims.d[i];
        shape_flat *= out_dims.d[i];
        Log::error() << "out_dims.d[i] " << out_dims.d[i] << std::endl;
    }

    // output buffer
    std::vector<float> output_data(shape_flat);

    float* output = static_cast<float*>(buffers.getHostBuffer("conv2d_17"));
    for (int i=0; i<shape_flat; i++){
        output_data[i] = *(output+i);
    }

    return PredictionPtr(new Prediction(output_data, shape));

#endif

    Log::warning() << "NOT YET IMPLEMENTED.. WORK IN PROGRESS.." << std::endl;

    // ======================================================
}

Dims MLEngineTRT::Vector2Dims(std::vector<int>& vecdims){

    Dims dims;
    dims.nbDims = vecdims.size();
    for(int i=0; i<dims.nbDims; i++){
        dims.d[i] = vecdims[i];
    }

    return dims;
}
