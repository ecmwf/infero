#include <vector>
#include <string>

#include "eckit/testing/Test.h"

#include "infero/ml_engines/MLEngine.h"


using namespace eckit::testing;
using namespace infero;

namespace test {


CASE("ML Engine instantiation") {

    std::string choice("onnx");
    std::string path("/not-existing-path/");

    EXPECT_THROWS( MLEngine::create(choice, path) );
}




}  // namespace test


int main(int argc, char** argv) {
    return run_tests(argc, argv);
}
