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
#include "eckit/config/LocalConfiguration.h"

#ifdef HAVE_MPI
#include "eckit/mpi/Comm.h"
#endif

#include "infero/models/InferenceModel.h"

#ifdef HAVE_ONNX
#include "infero/models/InferenceModelONNX.h"
#endif

#ifdef HAVE_TF_C
#include "infero/models/InferenceModelTFC.h"
#endif

#ifdef HAVE_TFLITE
#include "infero/models/InferenceModelTFlite.h"
#endif

#ifdef HAVE_TENSORRT
#include "infero/models/InferenceModelTRT.h"
#endif

using namespace eckit;

namespace infero {

InferenceModel::InferenceModel(const eckit::Configuration& conf) : modelBuffer_{size_t(0)}{

}

InferenceModel::~InferenceModel() {
    if(isOpen_)
        close();
}

InferenceModel* InferenceModel::create(const std::string& type,
                                       const eckit::Configuration& conf)
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

#ifdef HAVE_TF_C
    if (type == "tf_c") {
        Log::info() << "creating RTEngineTFC.. " << std::endl;
        InferenceModel* ptr = new InferenceModelTFC(conf);
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

    throw BadValue("Engine type " + type + " not supported!", Here());
}

void InferenceModel::open()  {
    isOpen_ = true;
}

// inference for models with multiple inputs and outputs
void InferenceModel::infer_mimo(std::vector<eckit::linalg::TensorFloat*> &tIn, std::vector<const char*> &input_names,
                                std::vector<eckit::linalg::TensorFloat*> &tOut, std::vector<const char*> &output_names)
{
    // Take copy of the input tensors
    std::vector<eckit::linalg::TensorFloat*> inputTensors(tIn.begin(), tIn.end());

    // For each tensor that needs re-ordering, do it into a copy
    std::vector<std::unique_ptr<eckit::linalg::TensorFloat>> temporaryCopies;

    for (int i = 0; i < inputTensors.size(); ++i) {
        if (inputTensors[i]->isRight()) {

            Log::info() << i << "-th Input Tensor has right-layout, "
                        << "but left-layout is needed. Transforming to left.." << std::endl;
                        
            temporaryCopies.emplace_back(new eckit::linalg::TensorFloat(inputTensors[i]->transformRigthToLeftLayout()));
            inputTensors[i] = temporaryCopies.back().get();
        }
    }

    // do the actual inference..
    infer_mimo_impl(inputTensors, input_names, tOut, output_names);

}

// inference for models with multiple inputs and outputs
void InferenceModel::infer_mimo_impl(std::vector<eckit::linalg::TensorFloat*> &tIn, std::vector<const char*> &input_names,
                                     std::vector<eckit::linalg::TensorFloat*> &tOut, std::vector<const char*> &output_names)
{
    NOTIMP;
}

void InferenceModel::close() {
    isOpen_ = false;
}

void InferenceModel::broadcast_model(const std::string path) {

#ifdef HAVE_MPI
    modelBuffer_ = eckit::mpi::comm().broadcastFile(path, 0);
#endif
}

}  // namespace infero
