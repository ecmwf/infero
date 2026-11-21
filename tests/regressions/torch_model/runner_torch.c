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
#include <math.h>
#include "infero/api/infero.h"


void read_csv(const char* file_path, float* values){
    FILE *file = fopen(file_path, "r");
    double tmpval;
    int i;
    if ( file )
    {
       i=0;
       while ( fscanf(file, "%lf", &tmpval) == 1 )
       {
        *(values+i) = (float)tmpval;
        i++;
       }
       fclose(file);
    }
    else
    {
       perror(file_path);
    }
}

int main(int argc, char** argv){

    float tol = 1e-3;
    const int nInferenceReps = 10;

    char* model_path = argv[1];
    char* model_type = argv[2];
    if (argc>3) {
        tol = atof(argv[3]);
    }

    char yaml_str[1024];


    int input_size[1];
    size_t input_size_flatten;

    int output_size[1];
    size_t output_size_flatten;

    float* input_tensor;
    float* output_tensor;
    float* output_tensor_ref;

    infero_handle_t* infero_handle;

    assert(argc == 5);

    printf("model_path %s \n", model_path);
    printf("model_type %s \n", model_type);
    sprintf(yaml_str, " path: %s\n type: %s", model_path, model_type);
    printf("yaml_str:\n%s\n", yaml_str);

    // model input size [ 1, 10 ]
    input_size[0] = 10;
    input_size_flatten = input_size[0];
    input_tensor = (float*)malloc( sizeof (float) * input_size_flatten);

    // fill input tensor with 1.0
    for (int i=0; i<input_size_flatten; i++){
        *(input_tensor+i) = i;
    }

    // model output size [ 1, 1 ]
    output_size[0] = 1;
    output_size_flatten = output_size[0];
    output_tensor = (float*)malloc( sizeof (float) * output_size_flatten);

    // 0) init infero
    printf("initialising infero...\n");
    infero_initialise(argc, argv);

    // 1) get a inference model handle
    printf("creating handle...\n");
    infero_create_handle_from_yaml_str(yaml_str, &infero_handle);

    // 2) open the handle
    printf("opening handle...\n");
    infero_open_handle(infero_handle);

    // 3) run inference
    printf("running inference...\n");
    for (int i=0; i<nInferenceReps; i++){
        infero_inference_float( infero_handle,
                                1, input_tensor, input_size, 0,
                                1, output_tensor, output_size, 0 );
    }

    // 4) close and delete the handle
    infero_close_handle( infero_handle );
    infero_delete_handle( infero_handle );

    // 5) finalise
    infero_finalise(); 

    // check output
    printf("output[0] = %f \n", output_tensor[0]);

    output_tensor_ref = (float*)malloc( sizeof (float) * output_size_flatten);
    output_tensor_ref[0] = -0.27930790;

    for(int i=0; i<output_size_flatten; i++){
        if ( fabs(*(output_tensor+i) - *(output_tensor_ref+i)) > tol){
            printf("ERROR: output element %d (%f) is "
                   "different from expected value %f\n", i, *(output_tensor+i), *(output_tensor_ref+i) );
            exit(1);
        }
    }

    free(input_tensor);
    free(output_tensor);
    free(output_tensor_ref);

    return 0;
}

