
#include <algorithm>
#include <iterator>
#include <sstream>
#include <fstream>
#include <iostream>
#include <iomanip>

#include "cnpy/cnpy.h"
#include "eckit/log/Log.h"
#include "eckit/exception/Exceptions.h"

#include "MLTensor.h"
#define CSV_FLOAT_PRECISION 16


using namespace eckit;
using namespace eckit::linalg;


namespace infero {

MLTensor::MLTensor() :
    TensorFloat(),
    CurrentOrdering(Ordering::ROW_MAJOR)
{

}


MLTensor::MLTensor(const float *array, const std::vector<Size> &shape) :
    TensorFloat(array, shape),
    CurrentOrdering(Ordering::ROW_MAJOR)
{

}


MLTensor::MLTensor(const std::vector<Tensor::Size> &shape) :
    TensorFloat(shape),
    CurrentOrdering(Ordering::ROW_MAJOR)
{

}


std::unique_ptr<MLTensor> MLTensor::copy_as(MLTensor::Ordering new_order) const
{

    std::unique_ptr<MLTensor> Tptr;
    Tptr = std::unique_ptr<MLTensor>(new MLTensor(*this));

    // shape cumulative
    std::vector<int> shape_cumul;
    shape_cumul.push_back(1);
    int tmp=1;
    for (int i=0; i<shape().size()-1; i++){
        tmp *= shape()[i];
        shape_cumul.push_back(tmp);
    }

    // shape cumulative reverse
    std::vector<int> shape_cumul_reverse(shape().size());
    tmp=1;
    for (int i=shape().size()-1; i>=1; i--){
        tmp *= shape()[i];
        shape_cumul_reverse[i-1] = tmp;
    }
    shape_cumul_reverse[shape().size()-1] = 1;


    if(CurrentOrdering==Ordering::ROW_MAJOR && new_order == Ordering::COL_MAJOR){

        std::vector<int> row_major_indexes(shape().size());
        int gidx_cm;
        for (int gidx_rm=0; gidx_rm<size(); gidx_rm++){

            // find the tensor indexes from the global index for a RM order
            row_major_indexes[0] = gidx_rm / shape_cumul_reverse[0];
            for (int idx=1; idx<row_major_indexes.size(); idx++){
                row_major_indexes[idx] = gidx_rm % shape_cumul_reverse[idx-1] / shape_cumul_reverse[idx];
            }

            // from the tensor indexes, work out the CM global index
            gidx_cm = 0;
            for (int idx=0; idx<row_major_indexes.size(); idx++){
                gidx_cm += row_major_indexes[idx] * shape_cumul[idx];
            }

            // assign the corresponding tensor value
            *(Tptr->data()+gidx_cm) = *(data()+gidx_rm);
        }

        // reset the ordering
        Tptr->CurrentOrdering = COL_MAJOR;


    // col-major to row-major
    } else if(CurrentOrdering==Ordering::COL_MAJOR && new_order == Ordering::ROW_MAJOR){

        std::vector<int> col_major_indexes(shape().size());

        int gidx_rm;
        for (int gidx_cm=0; gidx_cm<size(); gidx_cm++){

            // find the tensor indexes from the global index for a CM order
            for (int idx=0; idx<col_major_indexes.size()-1; idx++){
                col_major_indexes[idx] = (gidx_cm % shape_cumul[idx+1])/shape_cumul[idx];
            }
            col_major_indexes[col_major_indexes.size()-1] = gidx_cm / shape_cumul[shape_cumul.size()-1];

            // from the tensor indexes, work out the RM global index
            gidx_rm = 0;
            for(int idx=0; idx<col_major_indexes.size(); idx++){
                gidx_rm += col_major_indexes[idx] * shape_cumul_reverse[idx];
            }

            // assign the corresponding tensor value
            *(Tptr->data()+gidx_rm) = *(data()+gidx_cm);

        }

        // reset the ordering
        Tptr->CurrentOrdering = ROW_MAJOR;


    // no conversion required
    } else {

        // nothing to do
        ASSERT(CurrentOrdering == new_order);
    }

    return Tptr;

}


std::unique_ptr<MLTensor> MLTensor::from_file(const std::string &filename)
{
    std::unique_ptr<MLTensor> Tptr;

    std::string ext = filename.substr(filename.find_last_of("."));
    if (!ext.compare(".csv")){
        Tptr = from_csv(filename);
    } else if (!ext.compare(".npy")){
        Tptr = from_numpy(filename);
    } else {
        throw eckit::BadValue("File format "+ext+" not supported!", Here());
    }

    return Tptr;
}


void MLTensor::to_file(const std::string &filename)
{
    std::string ext = filename.substr(filename.find_last_of("."));
    if (!ext.compare(".csv")){
        to_csv(filename);
    } else if (!ext.compare(".npy")){
        to_numpy(filename);
    } else {
        throw eckit::BadValue("File format "+ext+" not supported!", Here());
    }

}


float MLTensor::compare(MLTensor &other, MLTensor::ErrorType mes) const
{

    ASSERT(size() == other.size());

    float err=0;

    switch(mes){

    case ErrorType::MSE:

        float val_tmp;
        for (int i=0; i<size(); i++){
            val_tmp = *(data()+i) - *(other.data()+i);
            err += val_tmp * val_tmp ;
        }
        err /= size();
        break;

    default:

        throw eckit::BadValue("Error measure not recognised!", Here());

    }

    return err;
}


std::unique_ptr<MLTensor> MLTensor::from_csv(const std::string &filename)
{
    Log::info() << "Reading CSV file " << filename << std::endl;

    // read nbdims
    std::ifstream file(filename);
    int ndims;
    file >> ndims;
    file.get();

    // read shape
    std::vector<size_t> local_shape(ndims);
    for (int i=0; i<ndims; i++){
        file >> local_shape[i];
        file.get();
    }

    // read data
    size_t sz=1;
    for (int i=0; i<ndims; i++){
        sz *= local_shape[i];
    }

    // fill the tensor (which has now ownership of allocated memory)
    auto tensor_ptr = std::unique_ptr<MLTensor>(new MLTensor(local_shape));
    for (int i=0; i<sz; i++){
        file >> *(tensor_ptr->data()+i);
        file.get();
    }

    file.close();

    return tensor_ptr;

}

void MLTensor::to_csv(const std::string &filename)
{

    Log::info() << "Writing CSV file " << filename << std::endl;

    std::ofstream of(filename);

    // dims
    of << shape().size() << ',';

    // shape
    for (const auto& d: shape()){
        of << d << ',';
    }

    // data
    for (size_t i = 0; i<size(); i++)
        of << std::setprecision(CSV_FLOAT_PRECISION) << *(data()+i) << ',';

    of.close();
    return;

}


std::unique_ptr<MLTensor> MLTensor::from_numpy(const std::string &filename)
{

    Log::info() << "Reading numpy file " << filename << std::endl;

    // read the numpy
    cnpy::NpyArray arr = cnpy::npy_load(filename);

    // shape
    std::vector<size_t> local_shape;
    local_shape.resize(arr.shape.size());
    for (size_t i = 0; i<arr.shape.size(); i++){
        local_shape[i] = arr.shape[i];
    }

    // fill the tensor (which has now ownership of allocated memory)
    auto tensor_ptr = std::unique_ptr<MLTensor>(new MLTensor(local_shape));

    std::vector<float> vv = arr.as_vec<float>();
    for (size_t i = 0; i<vv.size(); i++){
        *(tensor_ptr->data()+i) = vv[i];
    }

    return tensor_ptr;
}

void MLTensor::to_numpy(const std::string &filename)
{
    Log::info() << "Writing numpy file " << filename << std::endl;

    cnpy::npy_save(filename, data(), shape());
}


} // namespace infero
