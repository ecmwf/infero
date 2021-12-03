#include <stdlib.h>
#include <stdio.h>
#include "infero/api/infero.h"
#include "eckit/testing/Test.h"

using namespace eckit::testing;


CASE("infero_tensor_set_t"){

    char* arg1 = (char*)"arg1";
    char** args;
    args = &arg1;
    int argc = 1;
    int err;

    infero_tensor_set_t* h;
    
    EXPECT_EQUAL(infero_initialise(argc, args), INFERO_SUCCESS);

    err = infero_create_tensor_set(&h);
    EXPECT_EQUAL(err, INFERO_SUCCESS);

    int rank = 2;
    int shape[] = {2,5};
    float data[] = {0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9};
    const char* name = "test_tensor";

    err = infero_add_tensor(h, rank, shape, data, name, false);
    EXPECT_EQUAL(err, INFERO_SUCCESS);

    err = infero_print_tensor_set(h);
    EXPECT_EQUAL(err, INFERO_SUCCESS);

    EXPECT_EQUAL(infero_finalise(), INFERO_SUCCESS);
     
}


int main(int argc, char* argv[]) {
    return run_tests(argc, argv);
}