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
#include <cstdint>
#include <string>
#include <memory>

class Prediction;

typedef std::unique_ptr<Prediction> PredictionPtr;

// ML prediction
class Prediction
{

public:

    Prediction(std::vector<float> data_tensor,
               std::vector<int64_t> shape);

    // write output (format deduced by extension)
    int write_output(std::string filename);

    // save to disk as npy
    void to_numpy(std::string filename);

    // to CSV
    void to_csv(const std::string& filename,
                const std::string& delimiter=" ");

    // check prediction against reference file
    int verify_against(const std::string& filename,
                       const float& threshold = 0.01);


    // return the flat size
    int size_flat() const;

    // ptr to raw data
    float* data();

    int n_rows() const;

    int n_cols() const;

public:

    std::vector<float> data_tensor;

private:

    std::vector<int64_t> shape;
    std::vector<long unsigned int> shape_static;

};
