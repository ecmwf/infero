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

CASE("test_mltensor_row2col") {

    std::vector<size_t> shape{2, 3, 4};

    // ML Tensor is Row-major by default
    MLTensor t(shape);

    // now fill it with sequential values
    for (int i = 0; i < t.size(); i++) {
        *(t.data() + i) = i;
    }

    // expected internal values
    std::vector<size_t> ref_rm{
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
    };

    // check elements value by value
    for (int i = 0; i < t.size(); i++) {
        EXPECT(*(t.data() + i) == ref_rm[i]);
        //        std::cout << "*(t.data()+i) " << *(t.data()+i)
        //                  << ", ref_rm[i] " << ref_rm[i]
        //                     << std::endl;
    }

    // make a copy and store data in column-major format
    auto t2 = t.copy_as(MLTensor::COL_MAJOR);
    EXPECT(t2->size() == t.size());

    // expected internal values
    std::vector<size_t> ref_cm{0, 12, 4, 16, 8, 20, 1, 13, 5, 17, 9, 21, 2, 14, 6, 18, 10, 22, 3, 15, 7, 19, 11, 23};

    // check elements value by value
    for (int i = 0; i < t2->size(); i++) {
        EXPECT(*(t2->data() + i) == ref_cm[i]);
        //        std::cout << "*(t2->data()+i) " << *(t2->data()+i)
        //                  << ", ref_cm[i] " << ref_cm[i]
        //                     << std::endl;
    }
}


CASE("test_mltensor_col2row") {

    std::vector<size_t> shape{2, 3, 4};

    // ML Tensor is Row-major by default
    MLTensor t(shape);
    t.fill(0);

    // make a copy and store data in col-major format
    auto t2 = t.copy_as(MLTensor::COL_MAJOR);
    EXPECT(t2->size() == t.size());

    // now fill it with sequential values
    for (int i = 0; i < t.size(); i++) {
        *(t2->data() + i) = i;
        //        std::cout << "*(t2->data()+i) " << *(t2->data()+i) << std::endl;
    }

    // and make a copy back in row-major format (so that the indexing is altered)
    auto t3 = t2->copy_as(MLTensor::ROW_MAJOR);
    EXPECT(t3->size() == t2->size());

    // expected values
    std::vector<size_t> ref_rm{0, 6, 12, 18, 2, 8, 14, 20, 4, 10, 16, 22, 1, 7, 13, 19, 3, 9, 15, 21, 5, 11, 17, 23};

    // check elements value by value
    for (int i = 0; i < t3->size(); i++) {
        EXPECT(*(t3->data() + i) == ref_rm[i]);
        //        std::cout << "*(t3->data()+i) " << *(t3->data()+i)
        //                  << ", ref_rm[i] " << ref_rm[i]
        //                     << std::endl;
    }
}


}  // namespace test


int main(int argc, char** argv) {
    return run_tests(argc, argv);
}
