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


using namespace eckit;

namespace infero {

void NoOpDeallocator(void* data, size_t a, void* b) {
    // no input/output tensor deallocation here..
}


InferenceModelTFC::InferenceModelTFC(const eckit::Configuration& conf) :
    InferenceModel(conf) {

    std::string ModelPath(conf.getString("path"));

    // read/bcast model by mpi (when possible)
    broadcast_model(ModelPath);

    network_graph = TF_NewGraph();
    err_status = TF_NewStatus();
    session_options = TF_NewSessionOptions();
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


void InferenceModelTFC::infer(eckit::linalg::TensorFloat& tIn, eckit::linalg::TensorFloat& tOut) {

    // Input tensor (NB: implicitely assumed that we only have one input!)
    int NInputs = 1;

    // array of outputs of input-operations
    TF_Output* Input = static_cast<TF_Output*>(malloc(sizeof(TF_Output) * NInputs));

    // allocate and output of input-operation
    TF_Output t0 = {TF_GraphOperationByName(network_graph, "serving_default_input_1"), 0};
    INFERO_CHECK(t0.oper)

    Input[0] = t0;


    // Output tensor (NB: implicitely assumed that we only have one output!)
    int NOutputs = 1;

    // array of outputs of output-operations
    TF_Output* Output = static_cast<TF_Output*>(malloc(sizeof(TF_Output) * NOutputs));

    // allocate and output of output-operation
    TF_Output t2 = {TF_GraphOperationByName(network_graph, "StatefulPartitionedCall"), 0};
    INFERO_CHECK(t2.oper)

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

    if (tOut.isRight()) {

        // TFC uses Left (C) tensor layouts, so we need to convert
        TensorFloat tLeft(offsets, tOut.shape(), false);  // wrap data

        // creates temporary tensor with data in left layout
        tOut = tLeft.transformLeftToRightLayout();

    } else {

        // TFC uses Left (C) tensor layouts, so we can copy straight into memory of tOut
        Log::info() << "output size " << tOut.size() << std::endl;
        memcpy(tOut.data(), offsets, tOut.size() * sizeof(float));
    }

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

void InferenceModelTFC::broadcast_model(const std::string path){

  // Not available for this class, as TF_C needs to read a
  // whole directory rather than a single file

}

}  // namespace infero
