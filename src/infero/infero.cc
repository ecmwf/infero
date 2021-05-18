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

#include "eckit/option/CmdArgs.h"
#include "eckit/option/SimpleOption.h"
#include "eckit/runtime/Main.h"
#include "eckit/log/Log.h"
#include "eckit/serialisation/FileStream.h"

#include "infero/ml_engines/MLEngine.h"
#include "infero/MLTensor.h"


using namespace eckit;
using namespace eckit::option;
using namespace eckit::linalg;
using namespace infero;


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
    options.push_back(new SimpleOption<std::string>("ref_path", "Path to Reference prediction"));
    options.push_back(new SimpleOption<double>("threshold", "Verification threshold"));


    CmdArgs args(&usage, options, 0, 0, true);

    // input path
    std::string input_path = args.getString("input","data.npy");
    std::string model_path = args.getString("model", "model.onnx");
    std::string engine_type = args.getString("engine", "onnx");
    std::string output_path = args.getString("output", "out.csv");
    std::string ref_path = args.getString("ref_path", "");
    double threshold = args.getDouble("threshold", 0.001);

    // input data
    std::unique_ptr<infero::MLTensor> inputT = infero::MLTensor::from_file(input_path);

    // runtime engine
    std::unique_ptr<MLEngine> engine = MLEngine::create( engine_type, model_path);
    std::cout << *engine << std::endl;

    // Run inference
    std::unique_ptr<infero::MLTensor> predT = engine->infer(inputT);

    // save
    if (args.has("output")){
        predT->to_file(output_path);
    }

    // compare against ref values
    if (args.has("ref_path")){

        std::unique_ptr<infero::MLTensor> refT = infero::MLTensor::from_csv(ref_path);
        float err = predT->compare(*refT);
        Log::info() << "MSE error: " << err << std::endl;
        Log::info() << "threshold: " << threshold << std::endl;

        return !(err<threshold);
    }

    return EXIT_SUCCESS;
}