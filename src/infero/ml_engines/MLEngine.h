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

#include "infero/input_types/input_data.h"
#include "infero/Prediction.h"

#include <string>
#include <memory>

using namespace std;

class MLEngine;
typedef std::unique_ptr<MLEngine> RTEnginePtr;


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
    virtual PredictionPtr infer(InputDataPtr& input_sample) = 0;

    static RTEnginePtr create(std::string choice,
                              std::string model_path);

};

