#include <vector>

#include "eckit/log/Log.h"

#include "infero/ml_engines/engine.h"

#ifdef HAVE_ONNX
#include "infero/ml_engines/engine_onnx.h"
#endif

#ifdef HAVE_TFLITE
#include "infero/ml_engines/engine_tflite.h"
#endif

#ifdef HAVE_TRT
#include "infero/ml_engines/engine_trt.h"
#endif

using namespace eckit;


MLEngine::~MLEngine(){}

RTEnginePtr MLEngine::create(std::string choice,
                                           std::string model_path)
{
    Log::info() << "Loading model "
                << model_path
                << std::endl;

#ifdef HAVE_ONNX
    if (choice.compare("onnx") == 0){

        Log::info() << "creating RTEngineONNX.. "
                    << std::endl;

        return RTEnginePtr(new MLEngineONNX(model_path));
    }
#endif

#ifdef HAVE_TFLITE
    if (choice.compare("tflite") == 0){

        Log::info() << "creating RTEngineTFlite.. "
                    << std::endl;

        return RTEnginePtr(new MLEngineTFlite(model_path));
    }
#endif

#ifdef HAVE_TRT
    if (choice.compare("trt") == 0){

        Log::info() << "creating MLEngineTRT.. "
                    << std::endl;

        return RTEnginePtr(new MLEngineTRT(model_path));
    }
#endif

    Log::error() << "Engine type " << choice
                 << " not supported!"
                 << std::endl;

    return nullptr;

}
