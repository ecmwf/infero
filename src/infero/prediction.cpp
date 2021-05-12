/*
 * (C) Copyright 1996- ECMWF.
 * 
 * This software is licensed under the terms of the Apache Licence Version 2.0
 * which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
 * In applying this licence, ECMWF does not waive the privileges and immunities
 * granted to it by virtue of its status as an intergovernmental organisation
 * nor does it submit to any jurisdiction.
 */

#include "prediction.h"

#include <algorithm>

#include "cnpy/cnpy.h"

#include "eckit/log/Log.h"

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

    std::replace(shape_static.begin(), shape_static.end(), -1, 1);

}

void Prediction::to_numpy(std::string filename)
{

    cnpy::npy_save(filename, &(data_tensor[0]), shape_static);

    Log::info() << "Prediction written to " << filename << std::endl;

    for (const auto& i: shape_static)
        Log::info() << "shape_static " << i << std::endl;

}

int Prediction::size_flat() const {

  int _flat = 1;
  for (auto i : shape) {
    _flat *= i;
  }

  return _flat;
}

float *Prediction::data() { return data_tensor.data(); }

int Prediction::n_rows() const { return shape_static[1]; }

int Prediction::n_cols() const { return shape_static[2]; }
