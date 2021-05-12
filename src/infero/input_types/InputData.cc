/*
 * (C) Copyright 1996- ECMWF.
 * 
 * This software is licensed under the terms of the Apache Licence Version 2.0
 * which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
 * In applying this licence, ECMWF does not waive the privileges and immunities
 * granted to it by virtue of its status as an intergovernmental organisation
 * nor does it submit to any jurisdiction.
 */

#include "cnpy/cnpy.h"
#include "infero/input_types/InputData.h"
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
