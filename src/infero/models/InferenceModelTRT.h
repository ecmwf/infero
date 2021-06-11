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

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include <cuda_runtime_api.h> // needed ???
#include "NvInfer.h"
#include "argsParser.h"
#include "buffers.h"
#include "common.h"
#include "logger.h"
#include "parserOnnxConfig.h"

#include "infero/models/InferenceModel.h"


namespace infero {

class InferenceModelTRT : public InferenceModel {

    // short for infer ptr
    template <typename T>
    using SampleUniquePtr = std::unique_ptr<T, samplesCommon::InferDeleter>;

public:
    // TRT engine options
    struct TRTOptions {
        std::vector<int> InputDimsMin;
        std::vector<int> InputDimsMax;
        std::vector<int> InputDimsOpt;
        std::string input_layer_name;
        std::string output_layer_name;
        int workspace_size;
    };

public:
    InferenceModelTRT(const eckit::Configuration& conf);

    virtual ~InferenceModelTRT();

    static std::unique_ptr<InferenceModelTRT> from_onnx(std::string onnx_path, TRTOptions& options,
                                                  std::string trt_path = "model.trt");

protected:

    void infer(eckit::linalg::TensorFloat& tIn, eckit::linalg::TensorFloat& tOut);

    virtual void print(std::ostream& os) const;

private:

    class Logger : public ILogger {
        void log(Severity severity, const char* msg) throw() {
            // show info-level messages only
            if (severity == Severity::kINFO)
                std::cout << msg << std::endl;
        }
    };

    // utility converter std::vector to TRT Dims
    static Dims Vector2Dims(std::vector<int>& vecdims);    

private:
    nvinfer1::IRuntime* InferRuntime_;

    std::shared_ptr<nvinfer1::ICudaEngine> Engine_;  //!< The TensorRT engine used to run the network

    SampleUniquePtr<nvinfer1::INetworkDefinition> Network_;

    char* modelMem;
};

}  // namespace infero
