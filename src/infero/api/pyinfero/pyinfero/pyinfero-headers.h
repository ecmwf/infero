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

struct infero_handle_t;
typedef struct infero_handle_t infero_handle_t;

/**
 * initialize infero library
 */
int infero_initialise(int argc, char** argv);

/** 
 * Creates an ML engine handle from a YAML string 
 * */
int infero_create_handle_from_yaml_str(const char str[], infero_handle_t** h);

/** 
 * Creates an ML engine handle from a YAML file 
 * */
int infero_create_handle_from_yaml_file(const char path[], infero_handle_t** h);

/**
 * open a ML engine handle
 */
int infero_open_handle(infero_handle_t* h);

/**
 * close a ML engine handle
 */
int infero_close_handle(infero_handle_t* h);

/**
 * Destroys the ML engine handle
 */
int infero_delete_handle(infero_handle_t* h);

/**
* run a ML engine for inference (float)
*/
int infero_inference_float(infero_handle_t* h, 
                           int rank1, 
                           const float data1[], 
                           const int shape1[],
                           int iLayout,
                           int rank2,
                           float data2[], 
                           const int shape2[],
                           int oLayout);

/**
 * run a ML engine for inference (double)
 */
int infero_inference_double(infero_handle_t* h, 
                            int rank1, 
                            const double data1[], 
                            const int shape1[],
                            int iLayout,
                            int rank2,
                            double data2[], 
                            const int shape2[],
                            int oLayout);

/** run a ML engine for inference (float)
* with multi-input and multi-output
*/
int infero_inference_float_mimo(infero_handle_t* h,
                                int nInputs,
                                const char** iNames, 
                                const int* iRanks, 
                                const int** iShape, 
                                const float** iData,
                                int iLayout,
                                int nOutputs,
                                const char** oNames, 
                                const int* oRanks, 
                                const int** oShape, 
                                float** oData,
                                int oLayout);

/** run a ML engine for inference (float)
* with multi-input and multi-output
*/
int infero_inference_double_mimo(infero_handle_t* h,
                                int nInputs,
                                const char** iNames, 
                                const int* iRanks, 
                                const int** iShape, 
                                const double** iData,
                                int iLayout,
                                int nOutputs,
                                const char** oNames, 
                                const int* oRanks, 
                                const int** oShape, 
                                double** oData,
                                int oLayout);

/** 
 * Run mimo inference from tensor sets
 */
int infero_inference_float_map(infero_handle_t* h, void* imap, void* omap);

/** 
 * Run mimo inference from tensor sets
 */
int infero_inference_double_map(infero_handle_t* h, void* imap, void* omap);

/**
 * @brief infero_print_statistics
 * @param h: handle
 * @return
 */
int infero_print_statistics(infero_handle_t* h);

/**
 * @brief infero_print_config
 * @param h: handle
 * @return
 */
int infero_print_config(infero_handle_t* h);

/**
 * finalise the handle
 */
int infero_finalise();
