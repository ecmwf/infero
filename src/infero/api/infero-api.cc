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

using eckit::linalg::TensorDouble;
using eckit::linalg::TensorFloat;

void infero_inference_double(double data1[], int rank1, int shape1[], double data2[], int rank2, int shape2[]) {

    std::cout << "infero_inference_double()" << std::endl;

    TensorDouble T1(data1, shapify(rank1, shape1));
    TensorDouble T2(data2, shapify(rank2, shape2));

    std::cout << "T1 : " << T1 << std::endl;
    std::cout << "T2 : " << T2 << std::endl;
}

void infero_inference_float(float data1[], int rank1, int shape1[], float data2[], int rank2, int shape2[]) {

    std::cout << "infero_inference_float()" << std::endl;

    TensorFloat T1(data1, shapify(rank1, shape1));
    TensorFloat T2(data2, shapify(rank2, shape2));

    std::cout << "T1 : " << T1 << std::endl;
    std::cout << "T2 : " << T2 << std::endl;
}

#ifdef __cplusplus
}
#endif
