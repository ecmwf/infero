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
#include "eckit/mpi/Comm.h"

#include "infero/models/InferenceModel.h"

#ifdef HAVE_ONNX
#include "infero/models/InferenceModelONNX.h"
#endif

#ifdef HAVE_TFLITE
#include "infero/models/InferenceModelTFlite.h"
#endif

#ifdef HAVE_TENSORRT
#include "infero/models/InferenceModelTRT.h"
#endif

using namespace eckit;

namespace infero {

InferenceModel::InferenceModel(const eckit::Configuration& conf) {

#ifdef HAVE_MPI
    Log::info() << "mpi::comm().size() " << mpi::comm().size() << std::endl;
    Log::info() << "mpi::comm().rank() " << mpi::comm().rank() << std::endl;

    // Model configuration from CL
    std::string ModelPath(conf.getString("path"));

    model_buffer = nullptr;
    size_t model_buffer_size;
    char* model_buffer_data;

    // rank 0 reads data from disk
    if (mpi::comm().rank() == 0){
        model_buffer = InferenceModelBuffer::from_path(ModelPath);
        model_buffer_size = model_buffer->size();
        model_buffer_data = reinterpret_cast<char*>(model_buffer->data());
        Log::info() << "Rank 0 has read the model buffer. Broadcasting size.." << std::endl;
    }

    // rank 0 broadcasts model size
    mpi::comm().broadcast(model_buffer_size, 0);

    // all other ranks make space for data buffer
    if (mpi::comm().rank() != 0){
        model_buffer_data = new char[model_buffer_size];
    }

    // rank 0 broadcasts the model buffer
    mpi::comm().broadcast(model_buffer_data, model_buffer_data+model_buffer_size, 0);

    // all other ranks build their modelbuffer
    if (mpi::comm().rank() != 0){
        model_buffer = new InferenceModelBuffer(model_buffer_data, model_buffer_size);
    }

    Log::info() << "rank " << mpi::comm().rank()
                << " has buffer size " << model_buffer_size
                << std::endl;
#else
    // Model configuration from CL
    std::string ModelPath(conf.getString("path"));

    // rank 0 reads data from disk
    model_buffer = InferenceModelBuffer::from_path(ModelPath);

#endif

}

InferenceModel::~InferenceModel() {
    if(isOpen_)
        close();
}

InferenceModel* InferenceModel::create(const string& type,
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

void InferenceModel::close() {
    isOpen_ = false;
}

}  // namespace infero
