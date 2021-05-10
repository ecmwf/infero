#include "eckit/option/CmdArgs.h"
#include "eckit/option/SimpleOption.h"
#include "eckit/runtime/Main.h"
#include "eckit/log/Log.h"

#include <cuda_runtime_api.h>
#include "NvInfer.h"
#include "argsParser.h"
#include "buffers.h"
#include "common.h"
#include "logger.h"
#include "parserOnnxConfig.h"
#include "ml_engines/engine_trt.h"

#include <iostream>
#include <memory>

using namespace eckit;
using namespace eckit::option;

typedef SimpleOption<std::string> OptStr;


void usage(const std::string&){

    Log::info() << std::endl
                << "-------------------------------" << std::endl
                << "     Converter ONNX to TRT     " << std::endl
                << "-------------------------------" << std::endl
                << std::endl
                << "Reads an ONNX network and generates a TensorRT"
                   "Optimized model - The conversion must take place"
                   " in the machine where the inference is to be run."
                << std::endl;
}

int main(int argc, char** argv){

    Main::initialise(argc, argv);
    std::vector<Option*> options;

    std::string onnx_path;
    std::string trt_path;

    options.push_back(new OptStr("onnx_path",
                                 "Path to input ONNX file"));

    options.push_back(new OptStr("trt_path",
                                 "Path to output TRT file"));

    CmdArgs args(&usage, options, 0,0, true);

    // input path
    if (!args.has("onnx_path")){
        onnx_path = "model.onnx";
    } else {
        args.get("onnx_path", onnx_path);
    }

    // model path
    if (!args.has("trt_path")){
        trt_path = "model.trt";
    } else {
        args.get("trt_path", trt_path);
    }

    // WIP..

    return EXIT_SUCCESS;
}
