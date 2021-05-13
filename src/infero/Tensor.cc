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
#include <cmath>
#include <iomanip>
#include <limits>

#include "cnpy/cnpy.h"

#include "eckit/log/Log.h"

#include "eckit/exception/Exceptions.h"

using namespace eckit;

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


int Tensor::compare(const std::string& filename, float threshold) {

    std::ifstream ifile(filename);
    std::vector<float> other;
    float val;
    while (ifile >> val) {
        other.push_back(val);
    }
    ifile.close();

    ASSERT(other.size() == size());

    // verify against threshold
    float mean_rel_err = 0.0;
    for (int i=0; i<other.size(); i++){
        mean_rel_err += fabs( (other.data()[i] - data_[i]) / other[i] );
    }

    mean_rel_err /= other.size();

    int exit_code;
    if (mean_rel_err <= threshold){

        Log::info() << "Relative Error "
                    << mean_rel_err
                    << " LESS than threshold "
                    << threshold
                    << " => PASSED!"
                    << std::endl;

        exit_code = 0;

    } else {

        Log::info() << "Relative Error "
                    << mean_rel_err
                    << " GREATER than threshold "
                    << threshold
                    << " => FAILED!"
                    << std::endl;

        exit_code = 1;

    }

    return exit_code;
}

void Tensor::write(const std::string& filename) const {
    std::string ext = filename.substr(filename.find_last_of("."));
    if (!ext.compare(".csv")) {
        std::ofstream of(filename);
        for (auto& v : data_)
            of << v << ',';
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
