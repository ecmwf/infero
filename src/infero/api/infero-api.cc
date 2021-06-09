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

#include "infero/inference_models/InferenceModel.h"

#include "eckit/linalg/Tensor.h"
#include "eckit/config/YAMLConfiguration.h"


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
infero_model_handle infero_handle_open(char config_str[]) {

    eckit::YAMLConfiguration cfg(config_str);  // use config_str
    InferenceModel* model = InferenceModel::open(cfg.getString("type"), cfg);

    ASSERT(model);

    return model;
}


// close a ML engine handle
void infero_handle_close(infero_model_handle h) {
    ASSERT(h);
    InferenceModel* model = reinterpret_cast<InferenceModel*>(h);
    InferenceModel::close(model);
}


// run a ML engine for inference
void infero_inference_double(infero_model_handle h,
                             double data1[], int rank1, int shape1[],
                             double data2[], int rank2, int shape2[]) {

    std::cout << "infero_inference_double() - TO BE COMPLETED" << std::endl;

    ASSERT(h);
    InferenceModel* model = reinterpret_cast<InferenceModel*>(h);


    TensorDouble T1(data1, shapify(rank1, shape1));
    TensorDouble T2(data2, shapify(rank2, shape2));

    std::cout << "T1 : " << T1 << std::endl;
    std::cout << "T2 : " << T2 << std::endl;
}


// run a ML engine for inference
void infero_inference_float(infero_model_handle h,
                            float data1[], int rank1, int shape1[],
                            float data2[], int rank2, int shape2[]) {

    ASSERT(h);
    InferenceModel* model = reinterpret_cast<InferenceModel*>(h);

    std::cout << "infero_inference_float()" << std::endl;

    TensorFloat* tIn(new TensorFloat(data1, shapify(rank1, shape1), true));
    TensorFloat* tOut(new TensorFloat(data2, shapify(rank2, shape2), true));

    model->infer(*tIn, *tOut);

    // Needed as it is passed back to Fortran
    tOut->toRightLayout();
}

#ifdef __cplusplus
}
#endif
