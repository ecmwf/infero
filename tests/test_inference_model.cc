/*
 * (C) Copyright 1996- ECMWF.
 *
 * This software is licensed under the terms of the Apache Licence Version 2.0
 * which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
 * In applying this licence, ECMWF does not waive the privileges and immunities
 * granted to it by virtue of its status as an intergovernmental organisation
 * nor does it submit to any jurisdiction.
 */

#include <vector>
#include <string>

#include "eckit/testing/Test.h"
#include "eckit/config/LocalConfiguration.h"

#include "infero/models/InferenceModel.h"

using namespace eckit;
using namespace eckit::testing;
using namespace infero;

namespace test {


CASE("ML Engine instantiation") {

    std::string choice("onnx");
    std::string path("/not-existing-path/");

    // assemble model configuration
    LocalConfiguration local;
    {
        local.set("path", path);
    }
    const Configuration& conf = local;

    EXPECT_THROWS( InferenceModelFactory::instance().build(choice, conf) );
}




}  // namespace test


int main(int argc, char** argv) {
    return run_tests(argc, argv);
}
