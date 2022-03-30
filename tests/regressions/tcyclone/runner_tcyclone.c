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

    const float tol = 1e-5;

    char* model_path = argv[1];
    char* model_type = argv[2];
    char* input_path = argv[3];
    char* ref_output_path = argv[4];
    char yaml_str[1024];

    int input_size[4];
    size_t input_size_flatten;

    int output_size[4];
    size_t output_size_flatten;

    float* input_tensor;
    float* output_tensor;
    float* output_tensor_ref;

    infero_handle_t* infero_handle;

    assert(argc == 4);

    printf("model_path %s \n", model_path);
    printf("model_type %s \n", model_type);
    printf("input_path %s \n", input_path);
    printf("ref_output_path %s \n", ref_output_path);

    sprintf(yaml_str, " path: %s\n type: %s", model_path, model_type);
    printf("yaml_str:\n%s\n", yaml_str);

    // tcyclone model input size [ 1, 200, 200, 17 ]
    input_size[0] = 1;
    input_size[1] = 200;
    input_size[2] = 200;
    input_size[3] = 17;

    input_size_flatten = 1;
    for (int i=0; i<4; i++){
        input_size_flatten *= (size_t)input_size[i];
    }

    printf("input_size_flatten %li \n", input_size_flatten);
    input_tensor = (float*)malloc( sizeof (float) * input_size_flatten);

    // read input tensor..
    read_csv(input_path, input_tensor);

    // tcyclone model output size [ 1, 200, 200, 1 ]
    output_size[0] = 1;
    output_size[1] = 200;
    output_size[2] = 200;
    output_size[3] = 1;

    output_size_flatten = 1;
    for (int i=0; i<4; i++){
        output_size_flatten *= (size_t)output_size[i];
    }
    printf("output_size_flatten %li \n", output_size_flatten);
    output_tensor = (float*)malloc( sizeof (float) * output_size_flatten);

    // 0) init infero
    infero_initialise(argc, argv);

    // 1) get a inference model handle
    infero_create_handle_from_yaml_str(yaml_str, &infero_handle);

    // 2) open the handle
    infero_open_handle(infero_handle);

    // 3) run inference
    infero_inference_float_ctensor( infero_handle,
                                    4, input_tensor, input_size,
                                    4, output_tensor, output_size );

    // 4) close and delete the handle
    infero_close_handle( infero_handle );
    infero_delete_handle( infero_handle );

    // 5) finalise
    infero_finalise(); 

    // check output
    output_tensor_ref = (float*)malloc( sizeof (float) * output_size_flatten);
    read_csv(ref_output_path, output_tensor_ref);

    for(int i=0; i<output_size_flatten; i++){
        if ( *(output_tensor+i) - *(output_tensor_ref+i) > tol){
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

