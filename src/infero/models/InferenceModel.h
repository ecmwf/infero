/*
 * (C) Copyright 1996- ECMWF.
 *
 * This software is licensed under the terms of the Apache Licence Version 2.0
 * which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
 * In applying this licence, ECMWF does not waive the privileges and immunities
 * granted to it by virtue of its status as an intergovernmental organisation
 * nor does it submit to any jurisdiction.
 */

#pragma once

#include <memory>
#include <ostream>
#include <string>
#include <fstream>

#include "eckit/config/Configuration.h"
#include "eckit/linalg/Tensor.h"
#include "eckit/log/Log.h"

using eckit::Log;

namespace infero {

class InferenceModelBuffer;

/// Minimal interface for a inference model
class InferenceModel {


public:
    static InferenceModel* create(const std::string& type,
                                  const eckit::Configuration& conf,
                                  const InferenceModelBuffer* model_buffer = nullptr);

    InferenceModel();

    virtual ~InferenceModel();

    /// opens the engine
    virtual void open();

    /// run the inference
    virtual void infer(eckit::linalg::TensorFloat& tIn, eckit::linalg::TensorFloat& tOut) = 0;

    /// closes the engine
    virtual void close();

protected:
    /// print the model
    virtual void print(std::ostream& os) const = 0;

    friend std::ostream& operator<<(std::ostream& os, InferenceModel& obj) {
        obj.print(os);
        return os;
    }

private:
    bool isOpen_;
};


class InferenceModelBuffer
{
public:
    InferenceModelBuffer(void* data, size_t dataSize): data_(data), dataSize_(dataSize){}

    static InferenceModelBuffer* from_path(const std::string path){

        // read model from path
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        std::streamsize size = file.tellg();
        file.seekg(0, std::ios::beg);

        if (size <= 0)
            throw eckit::FailedSystemCall("File " + path +
                                          " has size " + std::to_string(size),
                                          Here());
        char* buffer = new char [size];
        if (file.read(buffer, size))
        {
            Log::info() << "Reading from " + path + " worked. " << std::endl;
            Log::info() << "Model size: " + std::to_string(size) << std::endl;
        }

        return new InferenceModelBuffer( reinterpret_cast<void*>(buffer),
                                         static_cast<size_t>(size));
    }

    void* data() const {return data_;}

    size_t size() const {return dataSize_;}

private:
    void* data_;
    size_t dataSize_;
};


}  // namespace infero
