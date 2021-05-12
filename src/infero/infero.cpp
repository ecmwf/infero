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

#include "infero/input_types/InputData.h"
#include "infero/ml_engines/MLEngine.h"
// #include "clustering/clustering.h"

#include "eckit/option/CmdArgs.h"
#include "eckit/option/SimpleOption.h"
#include "eckit/runtime/Main.h"
#include "eckit/log/Log.h"

#include <iostream>

using namespace eckit;
using namespace eckit::option;


void usage(const std::string&){

    Log::info() << std::endl
                << "-------------------------------" << std::endl
                << "Machine Learning execution tool" << std::endl
                << "-------------------------------" << std::endl
                << std::endl
                << "Reads a ML model + input data and "
                   "generates a prediction."
                << std::endl;
}


int main(int argc, char** argv) {

    Main::initialise(argc, argv);
    std::vector<Option*> options;

    options.push_back(new SimpleOption<std::string>("input", "Path to input file"));
    options.push_back(new SimpleOption<std::string>("output", "Path to output file"));
    options.push_back(new SimpleOption<std::string>("model", "Path to ML model"));
    options.push_back(new SimpleOption<std::string>("engine", "ML engine [onnx, tflite, trt]"));
    options.push_back(new SimpleOption<std::string>("clustering", "Clustering [dbscan, ...]"));
    options.push_back(new SimpleOption<std::string>("verify_against", "Path to Reference CSV values"));
    options.push_back(new SimpleOption<long>("verify_threshold", "Verification threshold"));


    CmdArgs args(&usage, options, 0, 0, true);

    // input path
    std::string input_path = args.getString("input","data.npy");
    std::string model_path = args.getString("model", "model.onnx");
    std::string engine_type = args.getString("engine", "onnx");
    std::string output_path = args.getString("output", "out.npy");
    std::string verify_against = args.getString("verify_against");
    float verify_threshold = args.getFloat("verify_threshold", 0.01);

    // input data
    InputDataPtr input_sample = InputData::from_numpy(input_path);
    if (!input_sample) {
        Log::error() << "Failed to read data from " << input_path << std::endl;
        return EXIT_FAILURE;
    }

    // runtime engine
    RTEnginePtr engine = MLEngine::create( engine_type, model_path);
    if (!engine) {
        Log::error() << "Failed to instantiate engine!" << std::endl;
        return EXIT_FAILURE;
    }

    // Run inference
    PredictionPtr prediction = engine->infer(input_sample);
    prediction->write_output(output_path);

    if (args.has("verify_against")){

        int exit_code;
        if (args.has("verify_threshold")){
            exit_code = prediction->verify_against(verify_against,
                                                   verify_threshold);
        } else {
            exit_code = prediction->verify_against(verify_against);
        }

        std::cout << "verify_threshold " << verify_threshold << std::endl;

        return exit_code;
    }

    // run clustering
    // std::string clustering_type = args.getString("clustering", "dbscan");
    // ClusteringPtr cluster = Clustering::create(clustering_type);
    // if (!cluster) {
    //     Log::error() << "Failed to instantiate clustering!" << std::endl;
    //     return EXIT_FAILURE;
    // }

    // int err = cluster->run(prediction);
    // if(err){
    //     Log::error() << "Clustering Failed!" << std::endl;
    //     return EXIT_FAILURE;
    // } else {
    //     // print prediction
    //     cluster->print_summary();

    //     // write to JSON
    //     cluster->write_json("out.json");
    // }

    return EXIT_SUCCESS;
}
