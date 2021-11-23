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

int main(int argc, char** argv){

    char* model_path = argv[1];
    char* model_type = argv[2];
    char* input_path = argv[3];
    char yaml_str[1024];

    int input_size[4];
    size_t input_size_flatten;
    double input_sum;

    int output_size[4];
    size_t output_size_flatten;
    float output_sum;

    float* input_tensor;
    float* output_tensor;
    double tmpval;
    int i;

    infero_handle_t* infero_handle;

    assert(argc == 4);

    printf("model_path %s \n", model_path);
    printf("model_type %s \n", model_type);
    printf("input_path %s \n", input_path);

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
    FILE *file = fopen(input_path, "r");
    input_sum = 0;
    if ( file )
    {
       i=0;       
       while ( fscanf(file, "%lf", &tmpval) == 1 )
       {
        input_tensor[i] = (float)tmpval;
        input_sum += tmpval;
        i++;
       }
       fclose(file);
    }
    else
    {
       perror(input_path);
    }

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

// //    // 3) run inference
// //    infero_inference_float_ctensor( infero_handle,
// //                                    input_tensor, 4, input_size,
// //                                    output_tensor, 4, output_size );

    // 4) close and delete the handle
    infero_close_handle( infero_handle );
    infero_delete_handle( infero_handle );

    // 5) finalise
    infero_finalise(); 


    // print sum of tensor output
    output_sum = 0;
    for (size_t i=0; i<output_size_flatten; i++){
        output_sum += output_tensor[i];
    }

    printf("input_sum %lf\n", input_sum);
    printf("output_sum %.8f\n", (double)output_sum);

    return 0;
}

