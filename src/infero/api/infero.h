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

/* Error handling */

/** Return codes */
enum InferoErrorValues
{
    INFERO_SUCCESS                 = 0,
    INFERO_ERROR_GENERAL_EXCEPTION = 1,
    INFERO_ERROR_UNKNOWN_EXCEPTION = 2
};

/** Returns a human-readable error message for the
 * last error given an error code
 * \param err Error code
 * \returns Error message
 */
const char* infero_error_string(int err);

/** Error handler callback function signature
 * \param context Error handler context
 * \param error_code Error code
 */
typedef void (*infero_failure_handler_t)(void* context, int error_code);

/** Sets an error handler which will be called on error
 * with the supplied context and an error code
 * \param handler Error handler function
 * \param context Error handler context
 *
 * To be called like so:
 * 
 *     void handle_failure(void* context, int error_code) {
 *        fprintf(stderr, "Error: %s\n", infero_error_string(error_code));
 *        clean_up();
 *        exit(1);
 *    }
 *    infero_set_failure_handler(handle_failure, NULL);
 */
int infero_set_failure_handler(infero_failure_handler_t handler, void* context);

// -------------------------------------------------------------

struct infero_tensors_t;
typedef struct infero_tensors_t infero_tensors_t;

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
 * run a ML engine for inference (double)
 */
int infero_inference_double(infero_handle_t* h, 
                            int rank1, 
                            const double data1[], 
                            const int shape1[], 
                            int rank2,
                            double data2[], 
                            const int shape2[]);

/** run a ML engine for inference (double)
 * for c-style input tensors
 */
int infero_inference_double_ctensor(infero_handle_t* h, 
                                    int rank1, 
                                    const double data1[], 
                                    const int shape1[], 
                                    int rank2,
                                    double data2[], 
                                    const int shape2[]);
/**
* run a ML engine for inference (float)
*/
int infero_inference_float(infero_handle_t* h, 
                           int rank1, 
                           const float data1[], 
                           const int shape1[], 
                           int rank2,
                           float data2[], 
                           const int shape2[]);


/** run a ML engine for inference (float)
 * for c-style input tensors
 */
int infero_inference_float_ctensor(infero_handle_t* h, 
                                   int rank1, 
                                   const float data1[], 
                                   const int shape1[], 
                                   int rank2,
                                   float data2[], 
                                   const int shape2[]);


/** run a ML engine for inference (float)
* with multi-input and multi-output
*/
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

/** run a ML engine for inference (float)
* with multi-input and multi-output - for C-style tensors
*/
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

/**
 * finalise the handle
 */
int infero_finalise();

#if defined(__cplusplus)
}  // extern "C"
#endif
