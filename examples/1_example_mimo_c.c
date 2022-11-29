/*
 * (C) Copyright 1996- ECMWF.
 *
 * This software is licensed under the terms of the Apache Licence Version 2.0
 * which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
 * In applying this licence, ECMWF does not waive the privileges and immunities
 * granted to it by virtue of its status as an intergovernmental organisation
 * nor does it submit to any jurisdiction.
 */
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <math.h>
#include "infero/api/infero.h"


/**
 * @brief Print tensors to stdout
 * 
 * @param n_tensors number of tensors to print
 * @param data pointer to data memory
 * @param names tensor names
 * @param shapes tensor shapes 
 * @param ranks tensor ranks
 */
void print_data(size_t n_tensors, float** data, char** names, int** shapes, int* ranks){

    // loop over tensors
    for (size_t t=0; t<n_tensors; t++){
        printf("--- tensor %s has rank %d\n", *(names+t), *(ranks+t));

        // loop over tensor axis
        int* shape = *(shapes+t);
        size_t n_values = 1;
        size_t shape_size = (size_t)(*(ranks+t));
        for (size_t s=0; s<shape_size; s++){
            printf("shape [%lu] %d \n", s, *(shape+s));
            n_values *= (size_t)(*(shape+s));
        }

        // loop over values
        printf("Values:\n");
        for (size_t v=0; v<n_values; v++){
            printf("value [%lu] %f \n", v, *(*(data+t)+v));
        }
    }
}


/**
 * @brief Utility function to delete memory allocated 
 * for input/output tensors
 * 
 * @param n_tensors number of tensors
 * @param data pointer to data memory
 * @param shapes tensor shapes
 */
void delete_data(size_t n_tensors, float** data, int** shapes){
    for (size_t t=0; t<n_tensors; t++){
        free(*(data+t));
        free(*(shapes+t));
    }
}


/**
 * @brief Small example that shows how to use the Infero C-API
 * for a multi-input Machine Learning model
 * 
 * @param argc N arguments
 * @param argv Arguments: model_path, model_type, name_input1, name_input2, name_output
 * @return int 
 */
int main(int argc, char** argv){

    char* model_path  = argv[1];
    char* model_type  = argv[2];
    char* name_input1 = argv[3];
    char* name_input2 = argv[4];
    char* name_output = argv[5];

    char yaml_str[1024];

    printf("model_path %s \n", model_path);
    printf("model_type %s \n", model_type);
    printf("name_input1 %s \n", name_input1);
    printf("name_input2 %s \n", name_input2);
    printf("name_output %s \n", name_output);

    // YAML configuration string
    // In the simplest case is as follows:
    //
    // path: <path/to/model>
    // type: <[onnx,tf_c,tensorrt, tflite]>
    //
    sprintf(yaml_str, " path: %s\n type: %s", model_path, model_type);
    printf("yaml_str:\n%s\n", yaml_str);

    // Here below we manually allocate and fill in 
    // the input arrays that will then be fed to the ML model
    // Input tensors are filled in row-wise as follows:
    
    // t1(1,:) = 0.1
    // t1(2,:) = 0.2
    // t1(3,:) = 0.3

    // t2(1,:) = 33.0
    // t2(2,:) = 66.0
    // t2(3,:) = 99.0

    // this is equivalent to the Fortran example (2_example_mimo_fortran)
    // and it shows how Infero handles ordering automatically,
    // both if an array comes from C or from Fortran.

    size_t n_inputs = 2;
    size_t batchSize = 3;

    float** inputs = malloc(sizeof(float*) * n_inputs);
    char** input_names = malloc(sizeof(const char*) * n_inputs);
    int** input_shapes = malloc(sizeof(int*) * n_inputs);
    int* iranks = malloc(sizeof(int) * n_inputs);

    // input 1 here below:
    // t1(1,:) = 0.1
    // t1(2,:) = 0.2
    // t1(3,:) = 0.3
    *(input_names) = name_input1;
    *(inputs) = (float*)malloc( sizeof (float) * batchSize * 32);
    for (size_t i=0; i<batchSize*32; i++){
        if(i>=0 && i<32*1){
          *(*(inputs)+i) = 0.1;
        } else if(i>=32*1 && i<32*2) {
          *(*(inputs)+i) = 0.2;  
        } else {
          *(*(inputs)+i) = 0.3;  
        }
    }
    *iranks = 2;
    *(input_shapes) = (int*)malloc( sizeof (int) * 2);
    *(*(input_shapes)) = batchSize;
    *(*(input_shapes)+1) = 32;

    // input 2 here below
    // t2(1,:) = 33.0
    // t2(2,:) = 66.0
    // t2(3,:) = 99.0    
    *(input_names+1) = name_input2;
    *(inputs+1) = (float*)malloc( sizeof (float) * batchSize * 128);
    for (size_t i=0; i<batchSize * 128; i++){
        if(i>=0 && i<128*1){
          *(*(inputs+1)+i) = 33.0;
        } else if(i>=128*1 && i<128*2) {
          *(*(inputs+1)+i) = 66.0;          
        } else {
          *(*(inputs+1)+i) = 99.0;  
        }
    }
    *(iranks+1) = 2;
    *(input_shapes+1) = (int*)malloc( sizeof (int) * 2);
    *(*(input_shapes+1)) = batchSize;
    *(*(input_shapes+1)+1) = 128;

    print_data(n_inputs, inputs, input_names, input_shapes, iranks);
    // ----------------------------------

    // Output tensors are allocated in this section
    size_t n_outputs = 1;
    float** outputs = malloc(sizeof(float*) * n_outputs);
    char** output_names = malloc(sizeof(const char*) * n_outputs);
    int** output_shapes = malloc(sizeof(int*) * n_outputs);
    int* oranks = malloc(sizeof(int) * n_outputs);

    *(output_names) = name_output;
    *(outputs) = (float*)malloc( sizeof (float) * batchSize * 1);

    *oranks = 2;
    *(output_shapes) = (int*)malloc( sizeof (int) * 2);
    *(*(output_shapes)) = batchSize;
    *(*(output_shapes)+1) = 1;

    print_data(n_outputs, outputs, output_names, output_shapes, oranks);
    // ----------------------------------

    // pointer to a Infero handle
    infero_handle_t* infero_handle;

    // 0) init infero library
    infero_initialise(argc, argv);

    // 1) create a inference model handle from the YAML configuration string
    infero_create_handle_from_yaml_str(yaml_str, &infero_handle);

    // 2) open the handle
    infero_open_handle(infero_handle);

    // 3) run inference
    infero_inference_float_mimo_ctensor(infero_handle,
                                        (int)n_inputs,
                                        (const char**)input_names,
                                        (const int*)iranks,
                                        (const int**)input_shapes,
                                        (const float**)inputs,
                                        (int)n_outputs,
                                        (const char**)output_names,
                                        (const int*)oranks,
                                        (const int**)output_shapes,
                                        outputs);

    // print output values
    print_data(n_outputs, outputs, output_names, output_shapes, oranks);

    // 4) close and delete the handle
    infero_close_handle( infero_handle );
    infero_delete_handle( infero_handle );

    // 5) finalise
    infero_finalise();

    // -------- delete data ----------
    delete_data(n_inputs, inputs, input_shapes);
    free(inputs);
    free(input_names);
    free(input_shapes);
    free(iranks);

    delete_data(n_outputs, outputs, output_shapes);
    free(outputs);
    free(output_names);
    free(output_shapes);
    free(oranks);
    // --------------------------------

}

