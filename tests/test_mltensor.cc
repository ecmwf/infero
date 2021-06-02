#include <vector>
#include "eckit/testing/Test.h"
#include "infero/MLTensor.h"

using namespace eckit::testing;
using namespace infero;

namespace test {

//-----------------------------------------------------------------------------

CASE("test_mltensor_row2row") {
    // TODO
}

CASE("test_mltensor_col2col") {
    // TODO
}

CASE("ML tensor [2,3,4] right to left layout conversion") {

    std::vector<size_t> shape{2, 3, 4};

    // ML Tensor is Right layout by default
    MLTensor t(shape);

    // now fill it with sequential values
    for (size_t i = 0; i < t.size(); i++) {
        *(t.data() + i) = i;
    }

    // expected internal values
    std::vector<size_t> ref_cm{0,1,2,3,4,5,6,7,8,9,10,11,12,13,
                               14,15,16,17,18,19,20,21,22,23,};

    // check elements value by value
    for (size_t i = 0; i < t.size(); i++) {
        EXPECT(*(t.data() + i) == ref_cm[i]);
        std::cout << "*(t.data()+i) " << *(t.data()+i)
                  << ", ref_cm[i] " << ref_cm[i]
                  << std::endl;
    }

    // to left layout
    t.toLeftLayout();

    // expected internal values
    std::vector<size_t> ref_rm{0,6,12,18,2,8,14,20,4,10,16,22,
                               1,7,13,19,3,9,15,21,5,11,17,23};

    // check elements value by value
    for (size_t i = 0; i < t.size(); i++) {
        EXPECT(*(t.data() + i) == ref_rm[i]);
        std::cout << "*(t2->data()+i) " << *(t.data()+i)
                  << ", ref_rm[i] " << ref_rm[i]
                  << std::endl;
    }
}




CASE("ML tensor [2,3,4] left to right layout conversion") {

    std::vector<size_t> shape{2, 3, 4};

    // ML Tensor is now Left
    MLTensor t(shape, false);

    // now fill it with sequential values
    for (size_t i = 0; i < t.size(); i++) {
        *(t.data() + i) = i;
    }

    // expected internal values
    std::vector<size_t> ref_rm{0,1,2,3,4,5,6,7,8,9,10,11,12,13,
                               14,15,16,17,18,19,20,21,22,23,};

    // check elements value by value
    for (size_t i = 0; i < t.size(); i++) {
        EXPECT(*(t.data() + i) == ref_rm[i]);
        std::cout << "*(t.data()+i) " << *(t.data()+i)
                  << ", ref_rm[i] " << ref_rm[i]
                  << std::endl;
    }

    // to right layout
    t.toRightLayout();

    // expected internal values
    std::vector<size_t> ref_cm{0,12,4,16,8,20,1,13,5,17,9,21,
                               2,14,6,18,10,22,3,15,7,19,11,23};

    // check elements value by value
    for (size_t i = 0; i < t.size(); i++) {
        EXPECT(*(t.data() + i) == ref_cm[i]);
        std::cout << "*(t2->data()+i) " << *(t.data()+i)
                  << ", ref_cm[i] " << ref_cm[i]
                  << std::endl;
    }
}

}  // namespace test


int main(int argc, char** argv) {
    return run_tests(argc, argv);
}
