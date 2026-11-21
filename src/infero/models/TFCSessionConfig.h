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
#include <string>
#include <vector>

#include "eckit/config/Configuration.h"

namespace infero {
namespace detail {

// The C API takes a serialized ConfigProto; avoid a dependency on the TensorFlow
// C++ protobuf headers. Field numbers come from tensorflow/core/protobuf/config.proto.
inline std::vector<uint8_t> makeTFCSessionConfig(const eckit::Configuration& config, int deviceID) {
    auto appendVarint = [](std::vector<uint8_t>& buffer, uint64_t value) {
        while (value >= 0x80) {
            buffer.push_back(static_cast<uint8_t>((value & 0x7f) | 0x80));
            value >>= 7;
        }
        buffer.push_back(static_cast<uint8_t>(value));
    };

    const std::string device = std::to_string(deviceID);
    std::vector<uint8_t> gpuOptions = {0x2a};  // GPUOptions.visible_device_list, field 5
    appendVarint(gpuOptions, device.size());
    gpuOptions.insert(gpuOptions.end(), device.begin(), device.end());

    // ConfigProto uses int32, not uint8_t. Negative int32 values must be
    // sign-extended to 64 bits and encoded as ten-byte varints.
    std::vector<uint8_t> buffer = {0x10};  // intra_op_parallelism_threads, field 2
    appendVarint(buffer, static_cast<uint64_t>(static_cast<int64_t>(config.getInt("numIntraopThreads"))));
    buffer.push_back(0x28);  // inter_op_parallelism_threads, field 5
    appendVarint(buffer, static_cast<uint64_t>(static_cast<int64_t>(config.getInt("numInteropThreads"))));
    buffer.push_back(0x32);  // gpu_options, field 6
    appendVarint(buffer, gpuOptions.size());
    buffer.insert(buffer.end(), gpuOptions.begin(), gpuOptions.end());
    return buffer;
}

}  // namespace detail
}  // namespace infero
