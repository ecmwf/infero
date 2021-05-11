#include "cnpy/cnpy.h"
#include "infero/input_types/input_data.h"
#include "eckit/log/Log.h"

using namespace eckit;



InputData::InputData(std::string filename){

    cnpy::NpyArray arr = cnpy::npy_load(filename);
    mDataVector = arr.as_vec<float>();

    Log::info() << "input shape: ";
    shape.resize(arr.shape.size());
    mSize = 1;
    for (size_t i = 0; i<arr.shape.size(); i++){
        shape[i] = arr.shape[i];
        mSize *= shape[i];
    }
}


InputDataPtr InputData::from_numpy(std::string filename)
{
    Log::info() << "Reading data from " << filename << std::endl;
    InputDataPtr sample(new InputData(filename));
    return sample;
}


float* InputData::get_data(){
    return mDataVector.data();
}


size_t InputData::get_size(){
    return mSize;
}
