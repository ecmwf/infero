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
#include <string>

#include "eckit/exception/Exceptions.h"
#include "eckit/log/Log.h"
#include "eckit/config/LocalConfiguration.h"

#include "infero/inference_models/InferenceModel.h"

#ifdef HAVE_ONNX
#include "infero/inference_models/InferenceModelONNX.h"
#endif

#ifdef HAVE_TFLITE
#include "infero/inference_models/InferenceModelTFlite.h"
#endif

#ifdef HAVE_TENSORRT
#include "infero/inference_models/InferenceModelTRT.h"
#endif

using namespace eckit;

namespace infero {

InferenceModel::InferenceModel() {}

InferenceModel::~InferenceModel() {
    if(isOpen_)
        close();
}

InferenceModel* InferenceModel::create(const string& type, const eckit::Configuration& conf) 
{
    std::string model_path(conf.getString("path"));
    Log::info() << "Loading model " << model_path << std::endl;

#ifdef HAVE_ONNX
    if (type == "onnx") {
        Log::info() << "creating RTEngineONNX.. " << std::endl;
        InferenceModel* ptr = new InferenceModelONNX(conf);
        return ptr;
    }
#endif

#ifdef HAVE_TFLITE
    if (type == "tflite") {
        Log::info() << "creating RTEngineTFlite.. " << std::endl;
        InferenceModel* ptr = new InferenceModelTFlite(conf);
        return ptr;
    }
#endif

#ifdef HAVE_TENSORRT
    if (type == "tensorrt") {
        Log::info() << "creating MLEngineTRT.. " << std::endl;
        InferenceModel* ptr = new InferenceModelTRT(conf);
        return ptr;
    }
#endif

    throw BadValue("Engine type " + choice + " not supported!", Here());
}

void InferenceModel::open()  {
    isOpen_ = true;
}

void InferenceModel::close() {
    isOpen_ = false;
}

}  // namespace infero
