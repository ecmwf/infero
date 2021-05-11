#pragma once

#include <string>

#include "infero/ml_engines/engine.h"


class MLEngineTFlite: public MLEngine
{

public:    

    MLEngineTFlite(std::string model_filename);

    virtual ~MLEngineTFlite();

    // build the engine
    virtual int build();

    // run the inference
    virtual PredictionPtr infer(InputDataPtr& input_sample);

};
