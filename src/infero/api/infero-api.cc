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

#ifdef __cplusplus
extern "C" {
#endif

void infero_infer_real64(double data1[], int rank1, int shape1[], double data2[], int rank2, int shape2[]) {
    std::cout << "infero_infer_real64()" << std::endl;
    for (int r = 0; r < rank1; ++r) {
        std::cout << "shape1[" << r << "] " << shape1[r] << std::endl;
    }
    for (int r = 0; r < rank2; ++r) {
        std::cout << "shape2[" << r << "] " << shape2[r] << std::endl;
    }
}

#ifdef __cplusplus
}
#endif
