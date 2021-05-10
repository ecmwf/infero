
#include "input_types/input_data.h"
#include "ml_engines/engine.h"
#include "clustering/clustering.h"

#include "eckit/option/CmdArgs.h"
#include "eckit/option/SimpleOption.h"
#include "eckit/runtime/Main.h"
#include "eckit/log/Log.h"

#include <memory>

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

typedef SimpleOption<std::string> OptStr;


int main(int argc, char** argv) {

    Main::initialise(argc, argv);
    std::vector<Option*> options;

    std::string input_path;
    std::string model_path;
    std::string engine_type;
    std::string clustering_type;

    options.push_back(new OptStr("input", "Path to input file"));
    options.push_back(new OptStr("model", "Path to ML model"));
    options.push_back(new OptStr("engine", "ML engine [onnx, tflite, trt]"));
    options.push_back(new OptStr("clustering", "Clustering [dbscan, ...]"));

    CmdArgs args(&usage, options, 0,0, true);

    // input path
    if (!args.has("input")){
        input_path = "data.npy";
    } else {
        args.get("input", input_path);
    }

    // model path
    if (!args.has("model")){
        model_path = "model.onnx";
    } else {
        args.get("model", model_path);
    }

    // engine type
    if (!args.has("engine")){
        engine_type = "onnx";
    } else {
        args.get("engine", engine_type);
    }

    // clustering type
    if (!args.has("clustering")){
        clustering_type = "dbscan";
    } else {
        args.get("clustering", clustering_type);
    }

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
    prediction->to_numpy("out.npy");

    // run clustering
    ClusteringPtr cluster = Clustering::create(clustering_type);
    if (!cluster) {
        Log::error() << "Failed to instantiate clustering!" << std::endl;
        return EXIT_FAILURE;
    }

    int err = cluster->run(prediction);
    if(err){
        Log::error() << "Clustering Failed!" << std::endl;
        return EXIT_FAILURE;
    } else {
        // print prediction
        cluster->print_summary();

        // write to JSON
        cluster->write_json("out.json");
    }

    Log::info() << "All done! " << std::endl;

    return EXIT_SUCCESS;
}
