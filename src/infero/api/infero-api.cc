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

#include "infero/ml_engines/MLEngine.h"
#include "infero/MLTensor.h"
#include "infero/utils.h"

#include "eckit/linalg/Tensor.h"


std::vector<size_t> shapify(int rank, int shape[]) {
    std::vector<size_t> result(rank);
    for (int i = 0; i < rank; ++i) {
        result[i] = shape[i];
    }
    return result;
}

//---------------------------------------------------------------------------------------------------


#ifdef __cplusplus
extern "C" {
#endif

using namespace std;
using namespace infero;

using eckit::linalg::TensorDouble;
using eckit::linalg::TensorFloat;


// open a ML engine handle
int infero_handle_open(char config_str[]){
    std::string str(config_str);
    trim(str);

    auto config = MLEngine::Configuration::from_yaml( str );
    int handle = MLEngine::create_handle(config.type(), config.path());

    return handle;
}


// close a ML engine handle
void infero_handle_close(int handle_id){
    MLEngine::close_handle(handle_id);
}


// run a ML engine for inference
void infero_inference_double(int handle_id,
                             double data1[], int rank1, int shape1[],
                             double data2[], int rank2, int shape2[]) {

    std::cout << "infero_inference_double()" << std::endl;

    TensorDouble T1(data1, shapify(rank1, shape1));
    TensorDouble T2(data2, shapify(rank2, shape2));

    std::cout << "T1 : " << T1 << std::endl;
    std::cout << "T2 : " << T2 << std::endl;
}


// run a ML engine for inference
void infero_inference_float(int handle_id,
                            float data1[], int rank1, int shape1[],
                            float data2[], int rank2, int shape2[]) {

    std::cout << "infero_inference_float()" << std::endl;

    std::unique_ptr<infero::MLTensor> T1ptr( new infero::MLTensor(data1, shapify(rank1, shape1)) );

    // from fortran to c-style
    T1ptr->toLeftLayout();

    std::unique_ptr<infero::MLTensor> T2ptr = MLEngine::get_model(handle_id)->infer(T1ptr);

    // from c-style to fortran
    T2ptr->toRightLayout();

    // TODO: this needs to be removed..
    memcpy(data2, T2ptr->data(), T2ptr->size() * sizeof(float) );
}

#ifdef __cplusplus
}
#endif
