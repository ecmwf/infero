#ifndef ENGINE_TFLITE_H
#define ENGINE_TFLITE_H

#include "ml_engines/engine.h"
#include <string>


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

#endif // TFLITE_H
