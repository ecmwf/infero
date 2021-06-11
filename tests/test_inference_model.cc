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

    EXPECT_THROWS( InferenceModel::create(choice, conf) );
}




}  // namespace test


int main(int argc, char** argv) {
    return run_tests(argc, argv);
}
