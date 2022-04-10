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


void delete_data(size_t n_tensors,
                 float** data,
                 int** shapes){

    for (size_t t=0; t<n_tensors; t++){
        free(*(data+t));
        free(*(shapes+t));
    }

}

int main(int argc, char** argv){

    const float tol = 1e-3;
    const int nInferenceReps = 10;

    char* model_path = argv[1];
    char* model_type = argv[2];
    char* name_input1 = argv[3];
    char* name_input2 = argv[4];
    char* name_output = argv[5];

    char yaml_str[1024];

    float expectedOutput[10] = {253.61697,
                                764.88446,
                                1276.1512,
                                1787.4171,
                                2298.686,
                                2809.9534,
                                3321.216,
                                3832.4849,
                                4343.7505,
                                4855.0225};

    printf("model_path %s \n", model_path);
    printf("model_type %s \n", model_type);
    printf("name_input1 %s \n", name_input1);
    printf("name_input2 %s \n", name_input2);
    printf("name_output %s \n", name_output);

    sprintf(yaml_str, " path: %s\n type: %s", model_path, model_type);
    printf("yaml_str:\n%s\n", yaml_str);

    // ------------ inputs --------------
    size_t n_inputs = 2;
    size_t batchSize = 10;

    float** inputs = malloc(sizeof(float*) * n_inputs);
    char** input_names = malloc(sizeof(const char*) * n_inputs);
    int** input_shapes = malloc(sizeof(int*) * n_inputs);
    int* iranks = malloc(sizeof(int) * n_inputs);

    // input 0
    *(input_names) = name_input1;
    *(inputs) = (float*)malloc( sizeof (float) * batchSize * 32);
    for (size_t i=0; i<batchSize*32; i++){
        *(*(inputs)+i) = ((float)i)/(batchSize*32);
    }
    *iranks = 2;
    *(input_shapes) = (int*)malloc( sizeof (int) * 2);
    *(*(input_shapes)) = batchSize;
    *(*(input_shapes)+1) = 32;

    // input 1
    *(input_names+1) = name_input2;
    *(inputs+1) = (float*)malloc( sizeof (float) * batchSize * 128);
    for (size_t i=0; i<batchSize * 128; i++){
        *(*(inputs+1)+i) = ((float)i)/(batchSize * 128);
    }
    *(iranks+1) = 2;
    *(input_shapes+1) = (int*)malloc( sizeof (int) * 2);
    *(*(input_shapes+1)) = batchSize;
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
    *(outputs) = (float*)malloc( sizeof (float) * batchSize * 1);

    *oranks = 2;
    *(output_shapes) = (int*)malloc( sizeof (int) * 2);
    *(*(output_shapes)) = batchSize;
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
    for(int i=0; i<nInferenceReps; i++){

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
    }

    // print output
    print_data(n_outputs,
               outputs,
               output_names,
               output_shapes,
               oranks);

    // take the output value
    float res = *(*outputs);

    for(int i=0; i<batchSize; i++){
        if (*(*(outputs)+i)-(*(expectedOutput+i)) > tol){
            printf("ERROR: output element %d (%f) is "
                   "different from expected value %f\n", i, *(*(outputs)+i), (*(expectedOutput+i)) );
            exit(1);
        }
    }

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

}

