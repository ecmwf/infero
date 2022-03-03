/*
 * (C) Copyright 1996- ECMWF.
 *
 * This software is licensed under the terms of the Apache Licence Version 2.0
 * which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
 * In applying this licence, ECMWF does not waive the privileges and immunities
 * granted to it by virtue of its status as an intergovernmental organisation
 * nor does it submit to any jurisdiction.
 */

enum InferoErrorValues
{
    INFERO_SUCCESS                 = 0,
    INFERO_ERROR_GENERAL_EXCEPTION = 1,
    INFERO_ERROR_UNKNOWN_EXCEPTION = 2
};

const char* infero_error_string(int err);

typedef void (*infero_failure_handler_t)(void* context, int error_code);

int infero_set_failure_handler(infero_failure_handler_t handler, void* context);

struct infero_tensor_set_t;

typedef struct infero_tensor_set_t infero_tensor_set_t;

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
                                const char** iNames,
                                const int* iRanks,
                                const int** iShape,
                                const float** iData,
                                int nOutputs,
                                const char** oNames,
                                const int* oRanks,
                                const int** oShape,
                                float** oData);

int infero_inference_float_mimo_ctensor(infero_handle_t* h,
                                        int nInputs,
                                        const char** iNames,
                                        const int* iRanks,
                                        const int** iShape,
                                        const float** iData,
                                        int nOutputs,
                                        const char** oNames,
                                        const int* oRanks,
                                        const int** oShape,
                                        float** oData);

int infero_inference_float_tensor_set(infero_handle_t* h,
                                      infero_tensor_set_t* iset,
                                      infero_tensor_set_t* oset);

int infero_print_statistics(infero_handle_t* h);

int infero_print_config(infero_handle_t* h);

int infero_finalise();

int infero_create_tensor_set(infero_tensor_set_t** h);

int infero_add_tensor(infero_tensor_set_t* h,
                      int rank,
                      int* shape,
                      float* data,
                      const char* name,
                      bool c_style
                      );

int infero_delete_tensor_set(infero_tensor_set_t* h);

int infero_print_tensor_set(infero_tensor_set_t* h);
