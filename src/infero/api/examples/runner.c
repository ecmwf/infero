/*
 * (C) Copyright 1996- ECMWF.
 *
 * This software is licensed under the terms of the Apache Licence Version 2.0
 * which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
 * In applying this licence, ECMWF does not waive the privileges and immunities
 * granted to it by virtue of its status as an intergovernmental organisation
 * nor does it submit to any jurisdiction.
 */

#include "infero.h"

void run_float() {
    int rank1              = 3;
    int shape1[3]          = {2, 2, 3};
    float data1[2 * 2 * 3] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0};

    int rank2          = 2;
    int shape2[2]      = {2, 3};
    float data2[2 * 3] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};

    //    infero_inference_float(data1, rank1, shape1, data2, rank2, shape2);

    // TODO: needs update
}

void run_double() {
    int rank1               = 3;
    int shape1[3]           = {2, 2, 3};
    double data1[2 * 2 * 3] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0};

    int rank2           = 2;
    int shape2[2]       = {2, 3};
    double data2[2 * 3] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};

    //    infero_inference_double(data1, rank1, shape1, data2, rank2, shape2);

    // TODO: needs update
}

int main() {
    run_float();
    run_double();
}
