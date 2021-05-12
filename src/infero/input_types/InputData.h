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

#include <vector>
#include <string>
#include <memory>


class InputData;

typedef std::unique_ptr<InputData> InputDataPtr;

class InputData
{

public:

    InputData(std::string filename);

    static InputDataPtr from_numpy(std::string filename);

    // get data
    float* get_data();

    // get size
    size_t get_size();

public:

    std::vector<size_t> shape;

private:

    size_t mSize;
    std::vector<float> mDataVector;

};
