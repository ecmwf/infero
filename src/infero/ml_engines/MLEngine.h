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

#include <string>
#include <memory>

#include "infero/Tensor.h"


using namespace std;


// Minimal interface for a runtime engine
class MLEngine
{

protected:

    std::string mModelFilename;

public:

    MLEngine(std::string model_filename):
        mModelFilename(model_filename){
    }

    virtual ~MLEngine();

    // build the engine
    virtual int build() = 0;

    // run the inference
    virtual std::unique_ptr<Tensor> infer(std::unique_ptr<Tensor>& input_sample) = 0;

    static std::unique_ptr<MLEngine> create(std::string choice,
                                            std::string model_path);

};

