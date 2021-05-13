/*
 * (C) Copyright 1996- ECMWF.
 *
 * This software is licensed under the terms of the Apache Licence Version 2.0
 * which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
 * In applying this licence, ECMWF does not waive the privileges and immunities
 * granted to it by virtue of its status as an intergovernmental organisation
 * nor does it submit to any jurisdiction.
 */

#include "Tensor.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <cmath>
#include <iomanip>
#include <limits>

#include "cnpy/cnpy.h"

#include "eckit/log/Log.h"
#include "eckit/exception/Exceptions.h"
#include "eckit/serialisation/FileStream.h"

using namespace eckit;

#define CSV_FLOAT_PRECISION 16

Tensor::Tensor(const Tensor &other)
{
    // copy shape
    shape_.resize(other.nbDims());
    for (int i=0; i<other.nbDims(); i++){
        shape_[i] = other.shape(i);
    }

    // copy data
    data_.resize(other.size());
    for (int i=0; i<other.size(); i++){
        data_[i] = other.data()[i];
    }
}

Tensor::Tensor(std::vector<float> data, std::vector<int64_t> shape) :
    shape_(shape),
    data_(data)
{
    // check the shape matche the size of data
    size_t size = 1;
    for (auto i : shape_) {
        size *= i;
    }
    ASSERT(size == data_.size());

    // TODO: remove this assumption that the batch size is equal to 1
    std::replace(shape_.begin(), shape_.end(), -1, 1);
}

Tensor::Tensor(eckit::Stream &ss)
{
    // read nb dims
    size_t ndims;
    ss >> ndims;
    shape_.resize(ndims);

    // read shape
    for (int i=0; i<ndims; i++){
        ss >> shape_[i];
    }

    // read data
    size_t sz=1;
    for (int i=0; i<ndims; i++){
        sz *= shape_[i];
    }
    data_.resize(sz);
    ss.readBlob(const_cast<float*>(&data_[0]), sizeof(float) * sz);
}


Tensor::Comparison Tensor::compare(const Tensor& other, float threshold) {

    ASSERT(other.size() == size());

    // verify against threshold
    float mean_rel_err = 0.0;
    for (int i=0; i<other.size(); i++){
        mean_rel_err += fabs( ( *(other.data()+i) - data_[i]) / *(other.data()+i) );
    }

    mean_rel_err /= other.size();

    // result of comparison
    return Comparison(mean_rel_err, threshold, mean_rel_err <= threshold);
}

void Tensor::encode(eckit::Stream& ss) const
{
    ss << nbDims();
    for (const auto& d: shape_){
        ss << d;
    }
    ss.writeBlob(const_cast<float*>(data()), sizeof(float) * size());
}

std::unique_ptr<Tensor> Tensor::from_file(const std::string &filename)
{
    std::string ext = filename.substr(filename.find_last_of("."));

    std::vector<int64_t> local_shape;
    std::vector<float> local_data;

    if (!ext.compare(".csv")) {

        // read nbdims
        std::ifstream file(filename);
        int ndims;
        file >> ndims;
        file.get();

        // read shape
        local_shape.resize(ndims);
        for (int i=0; i<ndims; i++){
            file >> local_shape[i];
            file.get();
        }

        // read data
        size_t sz=1;
        for (int i=0; i<ndims; i++){
            sz *= local_shape[i];
        }

        local_data.resize(sz);
        for (int i=0; i<sz; i++){
            file >> local_data[i];
            file.get();
        }

        file.close();

    }
    else if (!ext.compare(".npy")) {

        cnpy::NpyArray arr = cnpy::npy_load(filename);
        local_data = arr.as_vec<float>();

        local_shape.resize(arr.shape.size());

        for (size_t i = 0; i<arr.shape.size(); i++){
            local_shape[i] = arr.shape[i];
        }

    } else {
        throw eckit::BadValue("File extension not recognised", Here());
    }

    return std::unique_ptr<Tensor>(new Tensor(local_data, local_shape));

}


void Tensor::write(const std::string& filename) const {

    std::string ext = filename.substr(filename.find_last_of("."));

    if (!ext.compare(".csv")) {
        std::ofstream of(filename);

        // dims
        of << nbDims() << ',';

        // shape
        for (const auto& d: shape_){
            of << d << ',';
        }

        // data
        for (auto& v : data_)
            of << std::setprecision(CSV_FLOAT_PRECISION) << v << ',';

        of.close();
        return;
    }
    else if (!ext.compare(".npy")) {
        std::vector<size_t> localshape(shape_.size());
        for (int i = 0; i < shape_.size(); ++i)
            localshape[i] = shape_[i];

        cnpy::npy_save(filename, &(data_[0]), localshape);
        return;
    }
    throw eckit::BadValue("File extension not recognised", Here());
}
