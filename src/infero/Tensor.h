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
#include <sstream>

#include "eckit/serialisation/Stream.h"


/// Dense tensor of numerical values
/// It does not implement striding of data
/// Ordering is RowMajor (C order), including in output

class Tensor {

public:

    /// Tensor Comparison info
    /// It account for the fact that Tensor comparison
    /// may include additional information to just true/false
    class Comparison {

    public:

        Comparison(float err, float thr, bool pass) :
            mean_rel_error(err),
            threshold(thr),
            passed(pass){}

        float RelError() const { return mean_rel_error;}
        float Threshold() const { return threshold;}
        bool  HasPassed() const { return passed;}
        bool  ExitCode() const { return !passed;}

        friend std::ostream& operator<<(std::ostream& ost,
                                        const Comparison& obj) {
            if (obj.HasPassed()){
                ost << "Relative Error " << obj.RelError()
                   << " LESS than threshold " << obj.Threshold()
                   << " => PASSED!";
            } else {
                ost << "Relative Error " << obj.RelError()
                   << " GREATER than threshold " << obj.Threshold()
                   << " => FAILED!";
            }

            return ost;
        }

    private:

        float mean_rel_error;
        float threshold;
        bool passed;
    };

public: // methods

    // -- Constructors

    // from another tensor
    Tensor(const Tensor& other);

    // from data and shape
    Tensor(std::vector<float> data_tensor, std::vector<int64_t> shape);

    // from Tensor stream
    Tensor(eckit::Stream& ss);


    // -- Mutators

    // compare against another tensor
    Comparison compare(const Tensor& other, float threshold = 0.01);

    // to stream
    void encode(eckit::Stream& ss) const;

    // from_file (only used for convenience)
    static std::unique_ptr<Tensor> from_file(const std::string& filename);


    // -- Accessors

    /// @returns flattened size of the whole tensor
    int size() const { return data_.size(); }

    const float* data() const { return data_.data(); }

    const std::vector<int64_t>& shape() const { return shape_; }
    int64_t shape(size_t idx) const { return shape_[idx]; }

    size_t nbDims() const { return shape_.size(); }

    void write(const std::string& filename) const;

protected: // methods

private: // members
    std::vector<int64_t> shape_;
    std::vector<float> data_;
};
