/*
 * (C) Copyright 1996- ECMWF.
 *
 * This software is licensed under the terms of the Apache Licence Version 2.0
 * which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
 * In applying this licence, ECMWF does not waive the privileges and immunities
 * granted to it by virtue of its status as an intergovernmental organisation
 * nor does it submit to any jurisdiction.
 */

#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

#include "eckit/config/LocalConfiguration.h"
#include "eckit/testing/Test.h"
#include "infero/models/TFCSessionConfig.h"

#ifdef HAVE_TF_C
#include "tensorflow/c/c_api.h"
#endif

using namespace eckit::testing;

namespace {

void expectConfig(int intra, int inter, int device, const std::vector<uint8_t>& expected) {
    eckit::LocalConfiguration config;
    config.set("numIntraopThreads", intra);
    config.set("numInteropThreads", inter);
    const auto buffer = infero::detail::makeTFCSessionConfig(config, device);
    EXPECT(buffer == expected);

#ifdef HAVE_TF_C
    // Check the real TensorFlow parser without loading a model or creating
    // thread pools/GPU devices (some cases intentionally use large counts).
    std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(TF_NewStatus(), TF_DeleteStatus);
    std::unique_ptr<TF_SessionOptions, decltype(&TF_DeleteSessionOptions)> options(
        TF_NewSessionOptions(), TF_DeleteSessionOptions);
    TF_SetConfig(options.get(), buffer.data(), buffer.size(), status.get());
    const auto code = TF_GetCode(status.get());
    EXPECT_EQUAL(code, TF_OK);
#endif
}

CASE("TFC session configuration keeps intra and inter thread settings distinct") {
    expectConfig(4, 2, 0, {0x10, 0x04, 0x28, 0x02, 0x32, 0x03, 0x2a, 0x01, '0'});
    expectConfig(2, 4, 9, {0x10, 0x02, 0x28, 0x04, 0x32, 0x03, 0x2a, 0x01, '9'});
}

CASE("TFC session configuration preserves automatic thread counts") {
    expectConfig(0, 0, 0, {0x10, 0x00, 0x28, 0x00, 0x32, 0x03, 0x2a, 0x01, '0'});
}

CASE("TFC session configuration encodes multibyte thread counts and device IDs") {
    expectConfig(127, 128, 10,
                 {0x10, 0x7f, 0x28, 0x80, 0x01, 0x32, 0x04, 0x2a, 0x02, '1', '0'});
    expectConfig(128, 255, 12,
                 {0x10, 0x80, 0x01, 0x28, 0xff, 0x01, 0x32, 0x04, 0x2a, 0x02, '1', '2'});
    expectConfig(256, 16384, 127,
                 {0x10, 0x80, 0x02, 0x28, 0x80, 0x80, 0x01, 0x32, 0x05, 0x2a, 0x03, '1', '2', '7'});
    expectConfig(16384, 256, 0,
                 {0x10, 0x80, 0x80, 0x01, 0x28, 0x80, 0x02, 0x32, 0x03, 0x2a, 0x01, '0'});
}

CASE("TFC session configuration preserves the full signed int32 thread range") {
    expectConfig(std::numeric_limits<int32_t>::max(), std::numeric_limits<int32_t>::max(), 0,
                 {0x10, 0xff, 0xff, 0xff, 0xff, 0x07,
                  0x28, 0xff, 0xff, 0xff, 0xff, 0x07, 0x32, 0x03, 0x2a, 0x01, '0'});
    // Negative inter_op_parallelism_threads selects the calling thread.
    expectConfig(1, -1, 0,
                 {0x10, 0x01, 0x28, 0xff, 0xff, 0xff, 0xff, 0xff,
                  0xff, 0xff, 0xff, 0xff, 0x01, 0x32, 0x03, 0x2a, 0x01, '0'});
    expectConfig(std::numeric_limits<int32_t>::min(), std::numeric_limits<int32_t>::min(), 0,
                 {0x10, 0x80, 0x80, 0x80, 0x80, 0xf8, 0xff, 0xff, 0xff, 0xff, 0x01,
                  0x28, 0x80, 0x80, 0x80, 0x80, 0xf8, 0xff, 0xff, 0xff, 0xff, 0x01,
                  0x32, 0x03, 0x2a, 0x01, '0'});
}

}  // namespace

int main(int argc, char** argv) {
    return run_tests(argc, argv);
}
