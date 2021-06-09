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


InferenceModel::~InferenceModel() {}

void InferenceModel::infer(TensorFloat &tIn, TensorFloat &tOut)
{
    // set the correct tensor layout
    set_input_layout(tIn);

    // call inference
    do_infer(tIn, tOut);
}

std::unique_ptr<InferenceModel> InferenceModel::create(std::string choice, std::string model_path) {

    Log::info() << "Loading model " << model_path << std::endl;

#ifdef HAVE_ONNX
    if (choice.compare("onnx") == 0) {
        Log::info() << "creating RTEngineONNX.. " << std::endl;
        return std::unique_ptr<InferenceModel>(new InferenceModelONNX(model_path));
    }
#endif

#ifdef HAVE_TFLITE
    if (choice.compare("tflite") == 0) {
        Log::info() << "creating RTEngineTFlite.. " << std::endl;
        return std::unique_ptr<InferenceModel>(new InferenceModelTFlite(model_path));
    }
#endif

#ifdef HAVE_TENSORRT
    if (choice.compare("tensorrt") == 0) {
        Log::info() << "creating MLEngineTRT.. " << std::endl;
        return std::unique_ptr<InferenceModel>(new InferenceModelTRT(model_path));
    }
#endif

    throw BadValue("Engine type " + choice + " not supported!", Here());
}


InferenceModel* InferenceModel::open(string choice, const eckit::Configuration& conf)
{

    std::string model_path(conf.getString("path"));
    Log::info() << "Loading model " << model_path << std::endl;

#ifdef HAVE_ONNX
    if (choice.compare("onnx") == 0) {
        Log::info() << "creating RTEngineONNX.. " << std::endl;
        InferenceModel* ptr = new InferenceModelONNX(model_path);
        return ptr;
    }
#endif

#ifdef HAVE_TFLITE
    if (choice.compare("tflite") == 0) {
        Log::info() << "creating RTEngineTFlite.. " << std::endl;
        InferenceModel* ptr = new InferenceModelTFlite(model_path);
        return ptr;
    }
#endif

#ifdef HAVE_TENSORRT
    if (choice.compare("tensorrt") == 0) {
        Log::info() << "creating MLEngineTRT.. " << std::endl;
        InferenceModel* ptr = new InferenceModelTFlite(model_path);
        return ptr;
    }
#endif

    throw BadValue("Engine type " + choice + " not supported!", Here());
}


void InferenceModel::close(InferenceModel *handle)
{
    delete handle;
}

}  // namespace infero
