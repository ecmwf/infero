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

#include "infero/ml_engines/MLEngine.h"


class MLEngineTFlite: public MLEngine
{

public:    

    MLEngineTFlite(std::string model_filename);

    virtual ~MLEngineTFlite();

    // build the engine
    virtual int build();

    // run the inference
    virtual std::unique_ptr<Tensor> infer(std::unique_ptr<Tensor>& input_sample);

};
