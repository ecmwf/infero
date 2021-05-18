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
#include <ostream>
#include <memory>

#include "infero/MLTensor.h"


using namespace std;
using namespace eckit::linalg;

namespace infero {

// Minimal interface for a runtime engine
class MLEngine
{

public:

    MLEngine(std::string model_filename):
        mModelFilename(model_filename){
    }

    virtual ~MLEngine();

    // run the inference
    virtual std::unique_ptr<infero::MLTensor> infer(std::unique_ptr<infero::MLTensor>& input_sample) = 0;

    // create concrete engines
    static std::unique_ptr<MLEngine> create(std::string choice, std::string model_path);

    friend std::ostream& operator<<(std::ostream& os, MLEngine& obj){
        obj.print(os);
        return os;
    }


protected:

    virtual void print(std::ostream& os) const {}

protected:

    std::string mModelFilename;

};

}
