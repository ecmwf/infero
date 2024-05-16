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

    float tol = 1e-2;
    const int nInferenceReps = 2;

    char* model_path = argv[1];
    char* model_type = argv[2];
    if (argc>3) {
        tol = atof(argv[3]);
    }

    char yaml_str[1024];

    float expectedOutput[2] = {0.6535, 0.5611};

    printf("model_path %s \n", model_path);
    printf("model_type %s \n", model_type);
    printf("test tolerance %f \n", tol);

    sprintf(yaml_str, " path: %s\n type: %s", model_path, model_type);
    printf("yaml_str:\n%s\n", yaml_str);

    // ------------ inputs --------------
    size_t n_inputs = 2;
    size_t input1_size = 10;
    size_t input2_size = 5;
    size_t batchSize = 1;

    float** inputs = malloc(sizeof(float*) * n_inputs);
    char** input_names = malloc(sizeof(const char*) * n_inputs);
    int** input_shapes = malloc(sizeof(int*) * n_inputs);
    int* iranks = malloc(sizeof(int) * n_inputs);

    // input 0
    *(input_names) = "input_1";
    *(inputs) = (float*)malloc( sizeof (float) * batchSize * input1_size);
    for (size_t i=0; i<batchSize*input1_size; i++){
        *(*(inputs)+i) = (float)i;
    }
    *iranks = 2;
    *(input_shapes) = (int*)malloc( sizeof (int) * 2);
    *(*(input_shapes)) = batchSize;
    *(*(input_shapes)+1) = input1_size;

    // input 1
    *(input_names+1) = "input_2";
    *(inputs+1) = (float*)malloc( sizeof (float) * batchSize * input2_size);
    for (size_t i=0; i<batchSize * input2_size; i++){
        *(*(inputs+1)+i) = (float)i;
    }
    *(iranks+1) = 2;
    *(input_shapes+1) = (int*)malloc( sizeof (int) * 2);
    *(*(input_shapes+1)) = batchSize;
    *(*(input_shapes+1)+1) = input2_size;

    print_data(n_inputs,
               inputs,
               input_names,
               input_shapes,
               iranks);
    // ----------------------------------

    // ------------ outputs -------------
    size_t n_outputs = 2;
    size_t output1_size = 1;
    size_t output2_size = 1;

    float** outputs = malloc(sizeof(float*) * n_outputs);
    char** output_names = malloc(sizeof(const char*) * n_outputs);
    int** output_shapes = malloc(sizeof(int*) * n_outputs);
    int* oranks = malloc(sizeof(int) * n_outputs);

    // output_1
    *(output_names) = "output_1";
    *(outputs) = (float*)malloc( sizeof (float) * batchSize * output1_size);
    *oranks = 2;
    *(output_shapes) = (int*)malloc( sizeof (int) * 2);
    *(*(output_shapes)) = batchSize;
    *(*(output_shapes)+1) = output1_size;

    // output_2
    *(output_names+1) = "output_2";
    *(outputs+1) = (float*)malloc( sizeof (float) * batchSize * output2_size);
    *(oranks+1) = 2;
    *(output_shapes+1) = (int*)malloc( sizeof (int) * 2);
    *(*(output_shapes+1)) = batchSize;
    *(*(output_shapes+1)+1) = output2_size;


    print_data(n_outputs, outputs, output_names, output_shapes, oranks);
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

        infero_inference_float_mimo(infero_handle,
                                    (int)n_inputs,
                                    (const char**)input_names,
                                    (const int*)iranks,
                                    (const int**)input_shapes,
                                    (const float**)inputs,
                                    0,
                                    (int)n_outputs,
                                    (const char**)output_names,
                                    (const int*)oranks,
                                    (const int**)output_shapes,
                                    outputs,
                                    0);
    }

    // print output
    print_data(n_outputs, outputs, output_names, output_shapes, oranks);

    // take the output value
    float res = *(*outputs);

    for(int i=0; i<batchSize; i++){
        if (fabs(*(*(outputs)+i)-(*(expectedOutput+i))) > tol){
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

