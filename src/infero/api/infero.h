/*
 * (C) Copyright 1996- ECMWF.
 *
 * This software is licensed under the terms of the Apache Licence Version 2.0
 * which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
 * In applying this licence, ECMWF does not waive the privileges and immunities
 * granted to it by virtue of its status as an intergovernmental organisation
 * nor does it submit to any jurisdiction.
 */

#if defined(__cplusplus)
extern "C" {
#endif


// open a ML engine handle
int infero_handle_open(char config_str[]);


// close a ML engine handle
void infero_handle_close(int handle_id);


// run a ML engine for inference (double)
void infero_inference_double(int handle_id,
                             double data1[], int rank1, int shape1[],
                             double data2[], int rank2, int shape2[]);


// run a ML engine for inference (float)
void infero_inference_float(int handle_id,
                            float data1[], int rank1, int shape1[],
                            float data2[], int rank2, int shape2[]);

#if defined(__cplusplus)
}  // extern "C"
#endif
