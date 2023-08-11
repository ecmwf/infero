/*
 * (C) Copyright 1996- ECMWF.
 *
 * This software is licensed under the terms of the Apache Licence Version 2.0
 * which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
 * In applying this licence, ECMWF does not waive the privileges and immunities
 * granted to it by virtue of its status as an intergovernmental organisation
 * nor does it submit to any jurisdiction.
 */

#include <memory>
#include <cstring>
#include <sstream>
#include <vector>
#include <thread>

#include "eckit/log/Log.h"
#include "eckit/runtime/Main.h"
#include "eckit/config/LocalConfiguration.h"

#include "infero/models/InferenceModel.h"
#include "infero/infero_utils.h"


using namespace infero;


void run_inference(std::shared_ptr<InferenceModel> engine,
                   std::map<std::string, eckit::linalg::TensorFloat*> input_map, 
                   std::map<std::string, eckit::linalg::TensorFloat*> output_map) {

    std::cout << "Running from thread " << std::this_thread::get_id() << std::endl;

    for (size_t iInfer=0; iInfer<20; iInfer++) {
      engine->infer_mimo(input_map, output_map);
    }

}



//// ==============================================================================================================
//// Example usage: 
//// > ./bin/4_example_mimo_thread $HOME/git/infero/tests/data/mimo_model/mimo_model.onnx onnx input_1 input_2 dense_6
//// ==============================================================================================================
int main(int argc, char** argv) {

    eckit::Main::initialise(argc, argv);

    if (argc<6) {
      printf("Error: This example must be invoked as: \n");  
      printf("<infero-build-path>/bin/4_example_mimo_thread ");
      printf("<infero-sources-path>/tests/data/mimo_model/mimo_model.onnx ");
      printf("onnx ");
      printf("input_1 ");
      printf("input_2 ");
      printf("dense_6\n");
      return 1;
    }      

    // Input path
    std::string model_path  = argv[1];
    std::string model_type  = argv[2]; 
    std::string name_input1 = argv[3];
    std::string name_input2 = argv[4];
    std::string name_output = argv[5];

    // Model configuration from CL
    eckit::LocalConfiguration local;
    local.set("path", model_path);
    local.set("type", model_type);

    // N batches
    size_t batchSize = 3;

    // Input 1 here below:
    // t1(1,:) = 0.1
    // t1(2,:) = 0.2
    // t1(3,:) = 0.3
    std::vector<size_t> input1_shape{batchSize,32};
    eckit::linalg::TensorFloat* t1{new eckit::linalg::TensorFloat(input1_shape, eckit::linalg::TensorFloat::Layout::RowMajor)};
    for (size_t i=0; i<batchSize * 32; i++){
        if(i<32*1){
          *(t1->data()+i) = 0.1;
        } else if(i>=32*1 && i<32*2) {
          *(t1->data()+i) = 0.2;  
        } else {
          *(t1->data()+i) = 0.3;  
        }
    }

    // Input 2 here below:
    // t2(1,:) = 33.0
    // t2(2,:) = 66.0
    // t2(3,:) = 99.0
    std::vector<size_t> input2_shape{batchSize,128};
    eckit::linalg::TensorFloat* t2{new eckit::linalg::TensorFloat(input2_shape, eckit::linalg::TensorFloat::Layout::ColMajor)};
    for (size_t i=0; i<batchSize * 128; i++){
        if(i<128*1){
          *(t2->data()+i) = 33.0;
        } else if(i>=128*1 && i<128*2) {
          *(t2->data()+i) = 66.0;          
        } else {
          *(t2->data()+i) = 99.0;  
        }
    }

    // Output tensor
    std::vector<size_t> out_shape_vec{batchSize, 1};
    eckit::linalg::TensorFloat* t3{new eckit::linalg::TensorFloat(out_shape_vec, eckit::linalg::TensorFloat::Layout::ColMajor)};

    // Inference model
    std::shared_ptr<InferenceModel> engine(InferenceModelFactory::instance().build(model_type, local));
    std::cout << *engine << std::endl;

    // pack input/output
    std::map<std::string, eckit::linalg::TensorFloat*> input_map;
    input_map.insert(make_pair(name_input1, t1));
    input_map.insert(make_pair(name_input2, t2));

    std::map<std::string, eckit::linalg::TensorFloat*> output_map;
    output_map.insert(make_pair(name_output, t3));


    size_t nThreads=10;
    std::vector<std::thread> threads;
    for (size_t iThread=0; iThread<nThreads; iThread++) {
      threads.push_back( std::thread( run_inference, engine, input_map, output_map ) );
    }


    // join all the threads
    for(auto& thread: threads) {
      if (thread.joinable()) {
        thread.join();
      }
    }

    // print output tensor
    std::cout << *t3 << std::endl;

    // delete allocated memory
    delete t1;
    delete t2;
    delete t3;

}
