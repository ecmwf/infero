/*
 * (C) Copyright 1996- ECMWF.
 * 
 * This software is licensed under the terms of the Apache Licence Version 2.0
 * which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
 * In applying this licence, ECMWF does not waive the privileges and immunities
 * granted to it by virtue of its status as an intergovernmental organisation
 * nor does it submit to any jurisdiction.
 */

#include <vector>

#include "eckit/log/Log.h"

#include "infero/ml_engines/MLEngine.h"

#ifdef HAVE_ONNX
#include "infero/ml_engines/MLEngineONNX.h"
#endif

#ifdef HAVE_TFLITE
#include "infero/ml_engines/MLEngineTFlite.h"
#endif

#ifdef HAVE_TRT
#include "infero/ml_engines/MLEngineTRT.h"
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
