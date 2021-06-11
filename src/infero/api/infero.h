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

typedef void* infero_model_handle;

// Creates an ML engine handle from a YAML string
infero_model_handle infero_create_handle_from_yaml_str(char str[]);

// Creates an ML engine handle from a YAML file
infero_model_handle infero_create_handle_from_yaml_file(char path[]);

// open a ML engine handle
void infero_open_handle(infero_model_handle);

// close a ML engine handle
void infero_close_handle(infero_model_handle);

// Destroys the ML engine handle
void infero_delete_handle(infero_model_handle);

// run a ML engine for inference (double)
void infero_inference_double(infero_model_handle h, double data1[], int rank1, int shape1[], double data2[], int rank2,
                             int shape2[]);


// run a ML engine for inference (float)
void infero_inference_float(infero_model_handle h, float data1[], int rank1, int shape1[], float data2[], int rank2,
                            int shape2[]);

#if defined(__cplusplus)
}  // extern "C"
#endif
