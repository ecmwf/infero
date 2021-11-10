/*
 * (C) Copyright 1996- ECMWF.
 *
 * This software is licensed under the terms of the Apache Licence Version 2.0
 * which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
 * In applying this licence, ECMWF does not waive the privileges and immunities
 * granted to it by virtue of its status as an intergovernmental organisation
 * nor does it submit to any jurisdiction.
 */

#include <iostream>
#include <vector>
#include <memory>

#include "infero/api/infero.h"
#include "infero/models/InferenceModel.h"

#include "eckit/config/YAMLConfiguration.h"
#include "eckit/runtime/Main.h"

#ifdef HAVE_MPI
  #include "eckit/io/SharedBuffer.h"
  #include "eckit/mpi/Comm.h"
#endif


//---------------------------------------------------------------------------------------------------

#ifdef __cplusplus
extern "C" {
#endif

using namespace std;
using namespace infero;

using eckit::linalg::TensorDouble;
using eckit::linalg::TensorFloat;


void infero_initialise(int argc, char** argv){
    eckit::Main::initialise(argc, argv);
}

infero_model_handle infero_create_handle_from_yaml_str(char str[]) {

    std::string str_(str);
    eckit::YAMLConfiguration cfg(str_);
    InferenceModel* model = InferenceModel::create(cfg.getString("type"), cfg);
    model->open();

    ASSERT(model);

    return model;
}

infero_model_handle infero_create_handle_from_yaml_file(char path[]) {

#ifdef HAVE_MPI
    eckit::SharedBuffer buff = eckit::mpi::comm().broadcastFile(path, 0);
    eckit::YAMLConfiguration cfg(buff);
#else
    eckit::YAMLConfiguration cfg(path);
#endif    

    InferenceModel* model = InferenceModel::create(cfg.getString("type"), cfg);

    ASSERT(model);

    return model;
}

void infero_open_handle(infero_model_handle h) {
    ASSERT(h);
    InferenceModel* model = reinterpret_cast<InferenceModel*>(h);
    model->open();
}


void infero_close_handle(infero_model_handle h) {
    ASSERT(h);
    InferenceModel* model = reinterpret_cast<InferenceModel*>(h);
    model->close();
}


void infero_delete_handle(infero_model_handle h) {
    ASSERT(h);
    InferenceModel* model = reinterpret_cast<InferenceModel*>(h);
    delete model;
    h = nullptr;
}


// run a ML engine for inference
void infero_inference_double(infero_model_handle h,
                             double data1[], int rank1, int shape1[],
                             double data2[], int rank2, int shape2[]) {

    NOTIMP;

    std::cout << "infero_inference_double() - TO BE COMPLETED" << std::endl;

    ASSERT(h);
    InferenceModel* model = reinterpret_cast<InferenceModel*>(h);


    TensorDouble T1(data1, eckit::linalg::shapify(rank1, shape1));
    TensorDouble T2(data2, eckit::linalg::shapify(rank2, shape2));

    std::cout << "T1 : " << T1 << std::endl;
    std::cout << "T2 : " << T2 << std::endl;
}


// run a ML engine for inference
void infero_inference_double_ctensor(infero_model_handle h,
                                     double data1[], int rank1, int shape1[],
                                     double data2[], int rank2, int shape2[]) {

    NOTIMP;

    std::cout << "infero_inference_double_ctensor() - used for c-style input tensors" << std::endl;

}


// run a ML engine for inference
void infero_inference_float(infero_model_handle h,
                            float data1[], int rank1, int shape1[],
                            float data2[], int rank2, int shape2[]) {

    ASSERT(h);
    InferenceModel* model = reinterpret_cast<InferenceModel*>(h);

    std::cout << "infero_inference_float()" << std::endl;

    TensorFloat* tIn(new TensorFloat(data1, eckit::linalg::shapify(rank1, shape1), true));
    TensorFloat* tOut(new TensorFloat(data2, eckit::linalg::shapify(rank2, shape2), true));

    model->infer(*tIn, *tOut);

    delete tIn;
    delete tOut;
}


// run a ML engine for inference
void infero_inference_float_mimo(infero_model_handle h,
                                 int nInputs,
                                 char** iNames, int* iRanks, int** iShape, float** iData,
                                 int nOutputs,
                                 char** oNames, int* oRanks, int** oShape, float** oData) {

    ASSERT(h);
    InferenceModel* model = reinterpret_cast<InferenceModel*>(h);

    std::cout << "infero_inference_float_mimo()" << std::endl;

    // loop over INPUT tensors
    ASSERT(nInputs >= 1);
    std::vector<TensorFloat*> inputData(static_cast<size_t>(nInputs));
    std::vector<char*>  inputNames(static_cast<size_t>(nInputs));
    for (size_t i=0; i<static_cast<size_t>(nInputs); i++){

        // rank
        size_t rank = static_cast<size_t>(*(iRanks+i));
        ASSERT(rank >= 1);

        // shape
        std::vector<size_t> shape_(rank);
        for (size_t rr=0; rr<rank; rr++){
            shape_[rr] = static_cast<size_t>(*(*(iShape+i)+rr));
        }

        // name and data
        inputNames[i] = *(iNames+i);
        inputData[i] = new TensorFloat(*(iData+i), shape_, true);
    }

    // loop over OUTPUT tensors
    ASSERT(nOutputs >= 1);
    std::vector<TensorFloat*> outputData(static_cast<size_t>(nOutputs));
    std::vector<char*>  outputNames(static_cast<size_t>(nOutputs));
    for (size_t i=0; i<static_cast<size_t>(nOutputs); i++){

        // rank
        size_t rank = static_cast<size_t>(*(oRanks+i));
        ASSERT(rank >= 1);

        // shape
        std::vector<size_t> shape_(rank);
        for (size_t rr=0; rr<rank; rr++){
            shape_[rr] = static_cast<size_t>(*(*(oShape+i)+rr));
        }

        // name and data
        outputNames[i] = *(oNames+i);
        outputData[i] = new TensorFloat(*(oData+i), shape_, true);
    }


    // mimo inference
    model->infer_mimo(inputData, inputNames, outputData, outputNames);

    // delete memory for input tensors
    for (size_t i=0; i<static_cast<size_t>(nInputs); i++){
        delete inputData[i];
    }

    // delete memory for output tensors
    for (size_t i=0; i<static_cast<size_t>(nOutputs); i++){
        delete outputData[i];
    }

}

// run a ML engine for inference
void infero_inference_float_ctensor(infero_model_handle h,
                                   float data1[], int rank1, int shape1[],
                                   float data2[], int rank2, int shape2[]) {

    ASSERT(h);
    InferenceModel* model = reinterpret_cast<InferenceModel*>(h);

    std::cout << "infero_inference_float_ctensor() - used for c-style input tensors" << std::endl;

    TensorFloat* tIn(new TensorFloat(data1, eckit::linalg::shapify(rank1, shape1), false));
    TensorFloat* tOut(new TensorFloat(data2, eckit::linalg::shapify(rank2, shape2), false));

    model->infer(*tIn, *tOut);

    delete tIn;
    delete tOut;
}

void infero_finalise(){
    // nothing to do here..
}

#ifdef __cplusplus
}
#endif
