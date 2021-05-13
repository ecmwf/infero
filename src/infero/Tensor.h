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

#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <iosfwd>


/// Dense tensor of numerical values
/// It does not implement striding of data
/// Ordering is RowMajor (C order), including in output

class Tensor {
public:

    Tensor(std::vector<float> data_tensor, std::vector<int64_t> shape);

    int compare(const std::string& filename, float threshold = 0.01);

    /// @returns flattened size of the whole tensor
    int size() const { return data_.size(); }

    float* data() { return data_.data(); }

    int64_t shape(size_t idx) { return shape_[idx]; }

    void write(const std::string& filename) const;

protected: // methods

private: // members
    std::vector<int64_t> shape_;
    std::vector<float> data_;
};
