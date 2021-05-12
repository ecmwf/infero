/*
 * (C) Copyright 1996- ECMWF.
 *
 * This software is licensed under the terms of the Apache Licence Version 2.0
 * which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
 * In applying this licence, ECMWF does not waive the privileges and immunities
 * granted to it by virtue of its status as an intergovernmental organisation
 * nor does it submit to any jurisdiction.
 */

#include <algorithm>
#include <fstream>
#include <cmath>
#include <iomanip>
#include <limits>

#include "cnpy/cnpy.h"

#include "eckit/log/Log.h"

#include "Prediction.h"


using namespace eckit;

Prediction::Prediction(std::vector<float> data_tensor,
                       std::vector<int64_t> shape):
    shape(shape),
    data_tensor(data_tensor)
{

    shape_static.resize(shape.size());
    for (int i=0; i<shape.size(); i++){
        shape_static[i] = shape[i];
    }

    // NB: here we make the assumption that the batch size is equal to 1
    std::replace(shape_static.begin(), shape_static.end(), -1, 1);

}

void Prediction::to_numpy(std::string filename)
{

    cnpy::npy_save(filename, &(data_tensor[0]), shape_static);

    Log::info() << "Prediction written to " << filename << std::endl;

    for (const auto& i: shape_static)
        Log::info() << "shape_static " << i << std::endl;

}

void Prediction::to_csv(const std::string& filename,
                        const std::string& delimiter)
{


    std::ofstream of(filename);
    int size = size_flat();
    for (int i=0; i<size; i++){
        of << data_tensor[i] << delimiter;
    }
    of.close();

    Log::info() << "Prediction written to "
                << filename << std::endl;

}

// try to fond the appropriate output method by
// file extension..
int Prediction::write_output(std::string filename)
{

    std::string ext = filename.substr(filename.find_last_of("."));

    if( !ext.compare(".csv") ){
        this->to_csv(filename);
    } else if ( !ext.compare(".npy") ){
        this->to_numpy(filename);
    } else {
        Log::error() << "Output file extension not supported!"
                     << std::endl;
        return -1;
    }

    return 0;
}

int Prediction::verify_against(const std::string& filename,
                               const float& threshold){

    // TODO read from a generic file, not just CSV..
    std::ifstream ifile(filename);
    std::vector<float> ref_vals;
    float val;
    while (ifile >> val){
        ref_vals.push_back(val);
    }
    ifile.close();

    // check that it is of the expected size
    assert(ref_vals.size() == size_flat());

    // verify against threshold
    float mean_rel_err = 0.0;
    for (int i=0; i<ref_vals.size(); i++){
        mean_rel_err += fabs( (ref_vals[i]-data_tensor[i]) / ref_vals[i] );
    }

    mean_rel_err /= ref_vals.size();

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

int Prediction::size_flat() const {

    int flat = 1;
    for (auto i : shape) {
        flat *= i;
    }

    return flat;
}

float *Prediction::data() { return data_tensor.data(); }

int Prediction::n_rows() const { return shape_static[1]; }

int Prediction::n_cols() const { return shape_static[2]; }
