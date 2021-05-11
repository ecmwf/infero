#pragma once

#include "infero/input_types/input_data.h"
#include "infero/prediction.h"

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

