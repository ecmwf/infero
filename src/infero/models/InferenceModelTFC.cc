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
#include <vector>

#include "eckit/log/Log.h"

#include "infero/models/InferenceModelTFC.h"
#include "infero/infero_utils.h"
#include "eckit/utils/StringTools.h"


using namespace eckit;

namespace infero {

static InferenceModelBuilder<InferenceModelTFC> tfcBuilder;


void NoOpDeallocator(void* data, size_t a, void* b) {
    // no input/output tensor deallocation here..
}

ModelParams_t InferenceModelTFC::implDefaultParams_(){
    ModelParams_t params_;
       
    params_["numInteropThreads"] = "1";
    params_["numIntraopThreads"] = "1";

    return params_;
}


InferenceModelTFC::InferenceModelTFC(const eckit::Configuration& conf) :
    InferenceModel(conf) {

    // Model configuration
    readConfig_(conf);

    std::string ModelPath(ModelConfig_->getString("path"));

    // read/bcast model by mpi (when possible)
    broadcast_model(ModelPath);

    network_graph = TF_NewGraph();
    err_status = TF_NewStatus();

    // options
    session_options = TF_NewSessionOptions();

    uint8_t numInteropThreads_ = ModelConfig_->getInt("numInteropThreads");
    uint8_t numIntraopThreads_ = ModelConfig_->getInt("numIntraopThreads");
    uint8_t buf[]={0x10,numInteropThreads_,0x28,numIntraopThreads_};
    TF_SetConfig(session_options, buf,sizeof(buf), err_status);    

    run_options = nullptr;

    // if not null, use the model buffer
    if (modelBuffer_.size()){

        Log::info() << "Constructing TFC model from buffer not implemented" << std::endl;

        NOTIMP;

    } else {  // otherwise construct from model path

        // default model serving tag
        const char* tags = "serve";
        int ntags = 1;

        session = TF_LoadSessionFromSavedModel(session_options,
                                               run_options,
                                               ModelPath.c_str(),
                                               &tags,
                                               ntags,
                                               network_graph,
                                               nullptr,
                                               err_status);

        check_status(err_status, "TF_LoadSessionFromSavedModel");
    }

}

InferenceModelTFC::~InferenceModelTFC() {

    // Free memory
    TF_DeleteGraph(network_graph);

    TF_DeleteSession(session, err_status);
    check_status(err_status, "TF_DeleteSession");

    TF_DeleteSessionOptions(session_options);
    TF_DeleteStatus(err_status);
}

std::string InferenceModelTFC::name() const
{
    return std::string(this->type());
}


void InferenceModelTFC::infer_impl(eckit::linalg::TensorFloat& tIn, eckit::linalg::TensorFloat& tOut,
                                   std::string input_name, std::string output_name) {

    // Input tensor
    int NInputs = 1;

    // array of outputs of input-operations
    TF_Output* Input = static_cast<TF_Output*>(malloc(sizeof(TF_Output) * NInputs));

    // Try to figure out input layer
    TF_Output t1 = GetInputOperationBuffer_(input_name);
    Input[0] = t1;

    // Output tensor (NB: implicitely assumed that we only have one output!)
    int NOutputs = 1;

    // array of outputs of output-operations
    TF_Output* Output = static_cast<TF_Output*>(malloc(sizeof(TF_Output) * NOutputs));

    // Try to figure out output layer
    TF_Output t2 = GetOutputOperationBuffer_(output_name);
    Output[0] = t2;

    // Allocate array of ptrs for input & output tensors
    TF_Tensor** InputValues = static_cast<TF_Tensor**>(malloc(sizeof(TF_Tensor*) * NInputs));
    TF_Tensor** OutputValues = static_cast<TF_Tensor**>(malloc(sizeof(TF_Tensor*) * NOutputs));

    // input rank and shape
    size_t InputNdims = tIn.shape().size();
    std::vector<int64_t> input_dims = utils::convert_shape<eckit::linalg::Size, int64_t>(tIn.shape());

    size_t InputSize = sizeof(float) * tIn.size();
    float* data = tIn.data();

    TF_Tensor* InputTensor = TF_NewTensor(TF_FLOAT,
                                          input_dims.data(),
                                          static_cast<int>(InputNdims),
                                          data,
                                          InputSize,
                                          &NoOpDeallocator,
                                          nullptr);
    INFERO_CHECK(InputTensor)

    InputValues[0] = InputTensor;

    // Run the Session
    TF_SessionRun(session, nullptr,
                  Input, InputValues, NInputs,
                  Output, OutputValues, NOutputs,
                  nullptr,
                  0,
                  nullptr,
                  err_status);

    check_status(err_status, "TF_SessionRun");

    int OutputNdims = TF_GraphGetTensorNumDims(network_graph, *Output, err_status);
    check_status(err_status, "TF_GraphGetTensorNumDims");

    Log::info() << "N output dims: " << OutputNdims << std::endl;

    int64_t* OutputDims = static_cast<int64_t*>(malloc(sizeof(int64_t) * OutputNdims));
    TF_GraphGetTensorShape(network_graph,
                           *Output,
                           OutputDims,
                           OutputNdims,
                           err_status);

    check_status(err_status, "TF_GraphGetTensorShape");
    INFERO_CHECK(OutputDims)

    for (int i=0; i<OutputNdims; i++){
        Log::info() << "N output dims: " << OutputDims[i] << std::endl;
    }

    // copy output data
    Log::info() << "Copying output..." << std::endl;
    void* buff = TF_TensorData(OutputValues[0]);
    float* offsets = static_cast<float*>(buff);

    eckit::Timing t_start(statistics_.timer());
    if (tOut.layout() == eckit::linalg::TensorFloat::Layout::ColMajor) {

        // TFC uses Left (C) tensor layouts, so we need to convert
        TensorFloat tLeft(offsets, tOut.shape(), false);  // wrap data        
        TensorFloat tRight = tLeft.transformRowMajorToColMajor();
        tOut = tRight;

    } else {

        // TFC uses Left (C) tensor layouts, so we can copy straight into memory of tOut
        Log::info() << "output size " << tOut.size() << std::endl;
        memcpy(tOut.data(), offsets, tOut.size() * sizeof(float));
    }
    statistics_.oTensorLayoutTiming_ += eckit::Timing{statistics_.timer()} - t_start;
}


void InferenceModelTFC::infer_mimo_impl(std::vector<eckit::linalg::TensorFloat*> &tIn, std::vector<const char*> &input_names,
                                        std::vector<eckit::linalg::TensorFloat*> &tOut, std::vector<const char*> &output_names)
{

    // N Input tensors
    size_t NInputs = input_names.size();

    // array of outputs for input-operations
    TF_Output* Input = static_cast<TF_Output*>(malloc(sizeof(TF_Output) * NInputs));
    for (size_t i=0; i<NInputs; i++){
        Input[i] = GetInputOperationBuffer_(input_names[i]);
    }

    // N Output tensors
    size_t NOutputs = output_names.size();

    // array of outputs for output-operations
    TF_Output* Output = static_cast<TF_Output*>(malloc(sizeof(TF_Output) * NOutputs));

    std::cout << "NOutputs: " << NOutputs << std::endl;
    for (size_t i=0; i<NOutputs; i++){
        Output[i] = GetOutputOperationBuffer_(output_names[i]);
    }

    // -----------------------------------------------
    // input tensors
    TF_Tensor** InputValues = static_cast<TF_Tensor**>(malloc(sizeof(TF_Tensor*) * NInputs));
    for (size_t i=0; i<NInputs; i++){
        InputValues[i] = TF_TensorFromData( tIn[i]->shape(), tIn[i]->data() );
    }

    // output tensors
    TF_Tensor** OutputValues = static_cast<TF_Tensor**>(malloc(sizeof(TF_Tensor*) * NOutputs));
    for (size_t i=0; i<NOutputs; i++){
        OutputValues[i] = TF_TensorFromData( tOut[i]->shape(), tOut[i]->data() );
    }
    // -----------------------------------------------

    // ------------ Run the Session ------------------
    TF_SessionRun(session,
                  nullptr,
                  Input,
                  InputValues,
                  static_cast<int>(NInputs),
                  Output,
                  OutputValues,
                  static_cast<int>(NOutputs),
                  nullptr,
                  0,
                  nullptr,
                  err_status);

    check_status(err_status, "TF_SessionRun");
    // -----------------------------------------------


    // --------------- copy output -------------------
    eckit::Timing t_start(statistics_.timer());
    for (size_t i=0; i<NOutputs; i++){

        void* buff = TF_TensorData(*(OutputValues+i));
        float* offsets = static_cast<float*>(buff);

        if (tOut[i]->layout() == eckit::linalg::TensorFloat::Layout::ColMajor) {

            Log::info() << i << "-th Output Tensor needs right-layout. "
                        << "Transforming left to right.." << std::endl;

            // TFC uses Left (C) tensor layouts, so we need to convert
            TensorFloat tLeft(offsets, tOut[i]->shape(), false);  // wrap data
            TensorFloat tRight = tLeft.transformRowMajorToColMajor();
            *tOut[i] = tRight;

        } else {

            // TFC uses Left (C) tensor layouts, so we can copy straight into memory of tOut
            memcpy(tOut[i]->data(), offsets, tOut[i]->size() * sizeof(float));
        }
    }
    statistics_.oTensorLayoutTiming_ += eckit::Timing{statistics_.timer()} - t_start;
    // -----------------------------------------------

    free(Input);
    free(Output);

    // delete input tensors
    for (size_t i=0; i<NInputs; i++){
        TF_DeleteTensor( InputValues[i]);
    }

    // delete output tensors
    for (size_t i=0; i<NOutputs; i++){
        TF_DeleteTensor( OutputValues[i]);
    }

    free(InputValues);
    free(OutputValues);

}

void InferenceModelTFC::print(std::ostream &os) const
{
    os << "A TFC Model" << std::endl;
}

void InferenceModelTFC::check_status(const TF_Status* s, std::string name){

    if(TF_GetCode(s) == TF_OK) {
        Log::info() << name << " OK" << std::endl;
    }
    else {
        Log::error() << name << " NOT OK" << std::endl;
        throw eckit::BadValue("Operation failed!", Here());
    }
}

TF_Output InferenceModelTFC::GetOperationBuffer_(std::string name, int op_id)
{

    // op output
    TF_Output t0{TF_GraphOperationByName(network_graph, name.c_str()), op_id};

    int t0_ndims = TF_GraphGetTensorNumDims(network_graph, t0, err_status);
    check_status(err_status, "TF_GraphGetTensorNumDims");
    Log::info() << "Layer " << name
                << " [id=" << op_id << "]"
                << " has " << t0_ndims
                << " dims." << std::endl;

    int64_t* t0_dims = static_cast<int64_t*>(malloc(sizeof(int64_t) * t0_ndims));
    TF_GraphGetTensorShape(network_graph,
                           t0,
                           t0_dims,
                           t0_ndims,
                           err_status);

    for (int i=0; i<t0_ndims; i++){
        Log::info() << "N output dims: " << t0_dims[i] << std::endl;
    }

    check_status(err_status, "TF_GraphGetTensorShape");

    free(t0_dims);

    return t0;

}

TF_Output InferenceModelTFC::GetInputOperationBuffer_(std::string name)
{

    /// TODO: this logic is overcomplicated, need to find a better way
    /// of getting input layer when no layer name is passed!
    std::string inputLayerName;

    if (name.empty()){

        size_t pos = 0;
        TF_Operation* input_oper;
        input_oper = TF_GraphNextOperation(network_graph, &pos);
        std::vector<std::string> inputLayVecStr = eckit::StringTools::split("/", TF_OperationName(input_oper));
        inputLayerName = "serving_default_"+inputLayVecStr[0]+"_input";
        TF_Output t1 = {TF_GraphOperationByName(network_graph, inputLayerName.c_str()), 0};

        if(!t1.oper){
            Log::info() << "Model input layer name : " << inputLayerName << " not valid, trying again.." << std::endl;
            inputLayerName = "serving_default_input_1";
            t1 = {TF_GraphOperationByName(network_graph, inputLayerName.c_str()), 0};
            if(!t1.oper){
                Log::error() << "Model input layer name : " << inputLayerName << " also not valid => aborting!" << std::endl;
                throw eckit::BadValue("Model input layer name could not be detected, "
                                      "Try assigning it through MIMO interface", Here());
            }
        }

    } else {
        inputLayerName = name;
    }
    Log::info() << "Input layer: " << inputLayerName << std::endl;

    // input tensor buffer
    TF_Output t1 = {TF_GraphOperationByName(network_graph, inputLayerName.c_str()), 0};
    INFERO_CHECK(t1.oper)

    return t1;
}

TF_Output InferenceModelTFC::GetOutputOperationBuffer_(std::string name)
{
    // output layer name (use default if empty)
    std::string outputLayerName;
    if (name.empty()){
        outputLayerName = "StatefulPartitionedCall";
    } else {
        outputLayerName = name;
    }

    // try to extract op ID from name
    std::vector<std::string> name_split = StringTools::split(":", outputLayerName);
    std::string name_text = name_split[0];

    // by default op id = 0, otherwise try "<name>:id"
    int op_id = 0;
    if(name_split.size() > 1){
        op_id = std::stoi(name_split[1]);
    }

    return GetOperationBuffer_(name_text, op_id);

}


TF_Tensor* InferenceModelTFC::TF_TensorFromData(const std::vector<size_t>& dims, float* data){

    size_t InputNdims = dims.size();

    int prod=1;
    for (auto& d: dims) {
       prod *= d;
    }
    size_t InputSize = sizeof(float) * prod;
    std::vector<int64_t> input_dims = utils::convert_shape<eckit::linalg::Size, int64_t>(dims);

    TF_Tensor* Tensor = TF_NewTensor(TF_FLOAT,
                                     input_dims.data(),
                                     static_cast<int>(InputNdims),
                                     data,
                                     InputSize,
                                     &NoOpDeallocator,
                                     nullptr);
    INFERO_CHECK(Tensor)

    return Tensor;

}



void InferenceModelTFC::broadcast_model(const std::string path){

  // Not available for this class, as TF_C needs to read a
  // whole directory rather than a single file

}

}  // namespace infero
