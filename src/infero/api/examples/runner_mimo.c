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



void print_data(size_t n_tensors,
                float** data,
                char** names,
                int** shapes,
                int* ranks){

    // loop over tensors
    for (size_t t=0; t<n_tensors; t++){
        printf("--- tensor %s has rank %d\n", *(names+t), *(ranks+t));

        // loop over tensor axis
        int* shape = *(shapes+t);
        size_t n_values = 1;
        size_t shape_size = (size_t)(*(ranks+t));
        for (size_t s=0; s<shape_size; s++){
            printf("shape [%li] %d \n", s, *(shape+s));
            n_values *= (size_t)(*(shape+s));
        }

        // loop over values
        printf("Values:\n");
        for (size_t v=0; v<n_values; v++){
            printf("value [%li] %f \n", v, *(*(data+t)+v));
        }

    }

}


void delete_data(size_t n_tensors,
                 float** data,
                 int** shapes){

    for (size_t t=0; t<n_tensors; t++){
        free(*(data+t));
        free(*(shapes+t));
    }

}

int main(int argc, char** argv){

    char* model_path = argv[1];
    char* model_type = argv[2];
    char* name_input1 = argv[3];
    char* name_input2 = argv[4];
    char* name_output = argv[5];
    char yaml_str[1024];

    printf("model_path %s \n", model_path);
    printf("model_type %s \n", model_type);
    printf("name_input1 %s \n", name_input1);
    printf("name_input2 %s \n", name_input2);
    printf("name_output %s \n", name_output);

    sprintf(yaml_str, " path: %s\n type: %s", model_path, model_type);
    printf("yaml_str:\n%s\n", yaml_str);

    // ------------ inputs --------------
    size_t n_inputs = 2;
    float** inputs = malloc(sizeof(float*) * n_inputs);
    char** input_names = malloc(sizeof(const char*) * n_inputs);
    int** input_shapes = malloc(sizeof(int*) * n_inputs);
    int* iranks = malloc(sizeof(int) * n_inputs);

    // input 0
    *(input_names) = name_input1;
    *(inputs) = (float*)malloc( sizeof (float) * 32);
    for (size_t i=0; i<32; i++){
        *(*(inputs)+i) = 1.;
    }
    *iranks = 2;
    *(input_shapes) = (int*)malloc( sizeof (int) * 2);
    *(*(input_shapes)) = 1;
    *(*(input_shapes)+1) = 32;

    // input 1
    *(input_names+1) = name_input2;
    *(inputs+1) = (float*)malloc( sizeof (float) * 128);
    for (size_t i=0; i<128; i++){
        *(*(inputs+1)+i) = 1.;
    }
    *(iranks+1) = 2;
    *(input_shapes+1) = (int*)malloc( sizeof (int) * 2);
    *(*(input_shapes+1)) = 1;
    *(*(input_shapes+1)+1) = 128;

    print_data(n_inputs,
               inputs,
               input_names,
               input_shapes,
               iranks);
    // ----------------------------------

    // ------------ outputs -------------
    size_t n_outputs = 1;
    float** outputs = malloc(sizeof(float*) * n_outputs);
    char** output_names = malloc(sizeof(const char*) * n_outputs);
    int** output_shapes = malloc(sizeof(int*) * n_outputs);
    int* oranks = malloc(sizeof(int) * n_outputs);

    *(output_names) = name_output;
    *(outputs) = (float*)malloc( sizeof (float) * 1);
    *(*(outputs)) = 1;
    *oranks = 2;
    *(output_shapes) = (int*)malloc( sizeof (int) * 1);
    *(*(output_shapes)) = 1;
    *(*(output_shapes)+1) = 1;

    print_data(n_outputs,
               outputs,
               output_names,
               output_shapes,
               oranks);
    // ----------------------------------

    infero_handle_t* infero_handle;

    // 0) init infero
    infero_initialise(argc, argv);

    // 1) get a inference model handle
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

    // print output
    print_data(n_outputs,
               outputs,
               output_names,
               output_shapes,
               oranks);

    // take the output value
    float res = *(*outputs);

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


    printf("all done. Res: %f\n", res);

    // check against expected value 5112.6704

    if (fabs((double)res-5112.6704) < 0.1){
        return 0;
    } else {
        return 1;
    };
}

