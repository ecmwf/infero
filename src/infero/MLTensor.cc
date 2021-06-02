
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <sstream>

#include "cnpy/cnpy.h"
#include "eckit/exception/Exceptions.h"
#include "eckit/log/Log.h"

#include "MLTensor.h"
#define CSV_FLOAT_PRECISION 16


using namespace eckit;
using namespace eckit::linalg;


namespace infero {

MLTensor::MLTensor(bool isRight) : TensorFloat(isRight){}


MLTensor::MLTensor(const float* array, const std::vector<Size>& shape, bool isRight) :
    TensorFloat(array, shape, isRight){}


MLTensor::MLTensor(const std::vector<Tensor::Size>& shape, bool isRight) : TensorFloat(shape, isRight){}


std::unique_ptr<MLTensor> MLTensor::from_file(const std::string& filename) {
    std::unique_ptr<MLTensor> Tptr;

    std::string ext = filename.substr(filename.find_last_of("."));
    if (!ext.compare(".csv")) {
        Tptr = from_csv(filename);
    }
    else if (!ext.compare(".npy")) {
        Tptr = from_numpy(filename);
    }
    else {
        throw eckit::BadValue("File format " + ext + " not supported!", Here());
    }

    return Tptr;
}


void MLTensor::to_file(const std::string& filename) {
    std::string ext = filename.substr(filename.find_last_of("."));
    if (!ext.compare(".csv")) {
        to_csv(filename);
    }
    else if (!ext.compare(".npy")) {
        to_numpy(filename);
    }
    else {
        throw eckit::BadValue("File format " + ext + " not supported!", Here());
    }
}


float MLTensor::compare(MLTensor& other, MLTensor::ErrorType mes) const {

    ASSERT(size() == other.size());

    float err = 0;

    switch (mes) {

        case ErrorType::MSE:

            float val_tmp;
            for (int i = 0; i < size(); i++) {
                val_tmp = *(data() + i) - *(other.data() + i);
                err += val_tmp * val_tmp;
            }
            err /= size();
            break;

        default:

            throw eckit::BadValue("Error measure not recognised!", Here());
    }

    return err;
}


std::unique_ptr<MLTensor> MLTensor::from_csv(const std::string& filename) {
    Log::info() << "Reading CSV file " << filename << std::endl;

    // read nbdims
    std::ifstream file(filename);
    int ndims;
    file >> ndims;
    file.get();

    // read shape
    std::vector<size_t> local_shape(ndims);
    for (int i = 0; i < ndims; i++) {
        file >> local_shape[i];
        file.get();
    }

    // read data
    size_t sz = 1;
    for (int i = 0; i < ndims; i++) {
        sz *= local_shape[i];
    }

    // fill the tensor (which has now ownership of allocated memory)
    auto tensor_ptr = std::unique_ptr<MLTensor>(new MLTensor(local_shape));
    for (int i = 0; i < sz; i++) {
        file >> *(tensor_ptr->data() + i);
        file.get();
    }

    file.close();

    return tensor_ptr;
}

void MLTensor::to_csv(const std::string& filename) {

    Log::info() << "Writing CSV file " << filename << std::endl;

    std::ofstream of(filename);

    // dims
    of << shape().size() << ',';

    // shape
    for (const auto& d : shape()) {
        of << d << ',';
    }

    // data
    for (size_t i = 0; i < size(); i++)
        of << std::setprecision(CSV_FLOAT_PRECISION) << *(data() + i) << ',';

    of.close();
    return;
}


std::unique_ptr<MLTensor> MLTensor::from_numpy(const std::string& filename) {

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
    auto tensor_ptr = std::unique_ptr<MLTensor>(new MLTensor(local_shape));

    std::vector<float> vv = arr.as_vec<float>();
    for (size_t i = 0; i < vv.size(); i++) {
        *(tensor_ptr->data() + i) = vv[i];
    }

    return tensor_ptr;
}

void MLTensor::to_numpy(const std::string& filename) {
    Log::info() << "Writing numpy file " << filename << std::endl;

    cnpy::npy_save(filename, data(), shape());
}


}  // namespace infero
