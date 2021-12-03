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

#include <fstream>
#include <iomanip>
#include <string>

#include "cnpy/cnpy.h"

#include "eckit/exception/Exceptions.h"
#include "eckit/linalg/Tensor.h"

#define CSV_FLOAT_PRECISION 16

#define INFERO_CHECK(x)                                            \
    if (!(x)) {                                                    \
        char err_str[1024];                                        \
        sprintf(err_str, "Error at %s:%d\n", __FILE__, __LINE__);  \
        throw AssertionFailed(err_str, Here());                    \
    }

using namespace eckit;
using namespace eckit::linalg;

namespace infero {
namespace utils {


// error types for tensor comparison
enum TensorErrorType
{
    MSE
};


/// useful to convert shape of size_t to/from sahpe of int64_t
template <typename F, typename T>
std::vector<T> convert_shape(const std::vector<F>& vec) {
    std::vector<T> vec_new;
    std::copy(vec.begin(), vec.end(), back_inserter(vec_new));
    return vec_new;
}


/// Tensor from properly formatted CSV file:
///
///  right-ness bool, rank, [tensor shape components], data...
///
template <typename S>
Tensor<S>* tensor_from_csv(const std::string& filename, bool isright = false) {

    Log::info() << "Reading Tensor CSV file " << filename << std::endl;

    std::ifstream file(filename);

    // right/left layout flag
    bool right;
    file >> right;
    file.get();

    // read nbdims
    size_t ndims;
    file >> ndims;
    file.get();

    // read shape components
    std::vector<size_t> local_shape(ndims);
    for (size_t i = 0; i < ndims; i++) {
        file >> local_shape[i];
        file.get();
    }

    // read data
    size_t sz = 1;
    for (size_t i = 0; i < ndims; i++) {
        sz *= local_shape[i];
    }

    // fill the tensor (which has now ownership of allocated memory)
    Tensor<S>* tensor_ptr = new Tensor<S>(local_shape, isright);
    for (size_t i = 0; i < sz; i++) {
        file >> *(tensor_ptr->data() + i);
        file.get();
    }

    file.close();

    return tensor_ptr;
}


/// write the tensor into a CSV file:
///
///  right-ness bool, rank, [tensor shape components], data...
///
template <typename S>
void tensor_to_csv(const Tensor<S>& T, const std::string& filename) {

    Log::info() << "Writing CSV file " << filename << std::endl;

    std::ofstream of(filename);

    // layout
    of << T.isRight() << ',';

    // dims
    of << T.shape().size() << ',';

    // shape
    for (const auto& d : T.shape()) {
        of << d << ',';
    }

    // data
    for (size_t i = 0; i < T.size(); i++)
        of << std::setprecision(CSV_FLOAT_PRECISION) << *(T.data() + i) << ',';

    of.close();
}


/// Tensor from numpy file .npy
template <typename S>
Tensor<S>* tensor_from_numpy(const std::string& filename, bool isright = false) {

    Log::info() << "Reading numpy file " << filename << std::endl;

    // read the numpy
    cnpy::NpyArray arr = cnpy::npy_load(filename);

    // shape
    std::vector<size_t> local_shape;
    local_shape.resize(arr.shape.size());
    for (size_t i = 0; i < arr.shape.size(); i++) {
        local_shape[i] = arr.shape[i];
    }

    // fill the tensor (which has now ownership of allocated memory)
    Tensor<S>* tensor_ptr = new Tensor<S>(local_shape, isright);

    std::vector<S> vv = arr.as_vec<S>();
    for (size_t i = 0; i < vv.size(); i++) {
        *(tensor_ptr->data() + i) = vv[i];
    }

    return tensor_ptr;
}


/// Tensor to numpy
template <typename S>
void tensor_to_numpy(const Tensor<S>& T, const std::string& filename) {
    Log::info() << "Writing numpy file " << filename << std::endl;
    cnpy::npy_save(filename, T.data(), T.shape());
}


template <typename S>
Tensor<S>* tensor_from_file(const std::string& filename, bool isright = false) {

    Tensor<S>* tensor_ptr;

    std::string ext = filename.substr(filename.find_last_of("."));
    if (ext == ".csv") {
        tensor_ptr = tensor_from_csv<S>(filename, isright);
    }
    else if (ext == ".npy") {
        tensor_ptr = tensor_from_numpy<S>(filename, isright);
    }
    else {
        throw eckit::BadValue("File format " + ext + " not supported!", Here());
    }

    return tensor_ptr;
}


template <typename S>
void tensor_to_file(const Tensor<S>& T, const std::string& filename) {
    std::string ext = filename.substr(filename.find_last_of("."));
    if (ext == ".csv") {
        tensor_to_csv<S>(T, filename);
    }
    else if (ext == ".npy") {
        tensor_to_numpy<S>(T, filename);
    }
    else {
        throw eckit::BadValue("File format " + ext + " not supported!", Here());
    }
}


template <typename S>
float compare_tensors(const Tensor<S>& T1, const Tensor<S>& T2, TensorErrorType mes) {

    ASSERT(T1.size() == T2.size());
    size_t size = T1.size();

    float err = 0;

    switch (mes) {

        case TensorErrorType::MSE:

            float val_tmp;
            for (size_t i = 0; i < size; i++) {
                val_tmp = *(T1.data() + i) - *(T2.data() + i);
                err += val_tmp * val_tmp;
            }
            err /= size;
            break;

        default:

            throw eckit::BadValue("Error measure not recognised!", Here());
    }

    return err;
}


}  // namespace utils
}  // namespace infero
