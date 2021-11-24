/*
 * (C) Copyright 1996- ECMWF.
 *
 * This software is licensed under the terms of the Apache Licence Version 2.0
 * which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
 * In applying this licence, ECMWF does not waive the privileges and immunities
 * granted to it by virtue of its status as an intergovernmental organisation
 * nor does it submit to any jurisdiction.
 */

/** Return codes */
enum InferoErrorValues
{
    INFERO_SUCCESS                 = 0,
    INFERO_ERROR_GENERAL_EXCEPTION = 1,
    INFERO_ERROR_UNKNOWN_EXCEPTION = 2
};

const char* infero_error_string(int err);
typedef void (*infero_failure_handler_t)(void* context, int error_code);
int infero_set_failure_handler(infero_failure_handler_t handler, void* context);

struct infero_tensors_t;
typedef struct infero_tensors_t infero_tensors_t;

struct infero_handle_t;
typedef struct infero_handle_t infero_handle_t;

int infero_initialise(int argc, char** argv);
int infero_create_handle_from_yaml_str(const char str[], infero_handle_t** h);
int infero_create_handle_from_yaml_file(const char path[], infero_handle_t** h);
int infero_open_handle(infero_handle_t* h);
int infero_close_handle(infero_handle_t* h);
int infero_delete_handle(infero_handle_t* h);
int infero_inference_double(infero_handle_t* h, 
                            int rank1, 
                            const double data1[], 
                            const int shape1[], 
                            int rank2,
                            double data2[], 
                            const int shape2[]);
int infero_inference_double_ctensor(infero_handle_t* h, 
                                    int rank1, 
                                    const double data1[], 
                                    const int shape1[], 
                                    int rank2,
                                    double data2[], 
                                    const int shape2[]);
int infero_inference_float(infero_handle_t* h, 
                           int rank1, 
                           const float data1[], 
                           const int shape1[], 
                           int rank2,
                           float data2[], 
                           const int shape2[]);
int infero_inference_float_ctensor(infero_handle_t* h, 
                                   int rank1, 
                                   const float data1[], 
                                   const int shape1[], 
                                   int rank2,
                                   float data2[], 
                                   const int shape2[]);
int infero_inference_float_mimo(infero_handle_t* h,
                                int nInputs,
                                char** const iNames, 
                                int* const iRanks, 
                                int** const iShape, 
                                float** const iData,
                                int nOutputs,
                                char** const oNames, 
                                int* const oRanks, 
                                int** const oShape, 
                                float** const oData);
int infero_inference_float_mimo_ctensor(infero_handle_t* h,
                                        int nInputs,
                                        char** const iNames, 
                                        int* const iRanks, 
                                        int** const iShape, 
                                        float** const iData,
                                        int nOutputs,
                                        char** const oNames, 
                                        int* const oRanks, 
                                        int** const oShape, 
                                        float** const oData);                                
int infero_finalise();