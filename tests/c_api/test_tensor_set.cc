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
    void* wrong_ptr_type;
    
    EXPECT_EQUAL(infero_initialise(argc, args), INFERO_SUCCESS);

    err = infero_create_tensor_set(&h);
    EXPECT_EQUAL(err, INFERO_SUCCESS);

    // correct tensor set
    int rank = 2;
    int shape[] = {2,5};
    float data[] = {0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9};
    const char* name = "test_tensor";

    err = infero_add_tensor(h, rank, shape, data, name, false);
    EXPECT_EQUAL(err, INFERO_SUCCESS);

    err = infero_print_tensor_set(h);
    EXPECT_EQUAL(err, INFERO_SUCCESS);

    // correct tensor initialisation
    int rank1 = 1;
    int shape1[] = {1};
    float data1[] = {1};    
    err = infero_add_tensor(h, rank1, shape1, data1, name, false);
    EXPECT_EQUAL(err, INFERO_SUCCESS);

    int rank2 = 3;
    int shape2[] = {1,2,3};
    float data2[] = {0, 1.1, 2.2, 3.3, 4.4, 5.5};
    err = infero_add_tensor(h, rank2, shape2, data2, name, false);
    EXPECT_EQUAL(err, INFERO_SUCCESS);

    // incorrect tensor initialisation
    int rank3 = -2;
    int shape3[] = {1,1};
    float data3[] = {0,1,2,3,4};
    err = infero_add_tensor(h, rank3, shape3, data3, name, false);
    EXPECT_EQUAL(err, INFERO_ERROR_GENERAL_EXCEPTION);

    int rank4 = 0;
    int shape4[] = {1,1};
    float data4[] = {0,1,2,3,4};
    err = infero_add_tensor(h, rank4, shape4, data4, name, false);
    EXPECT_EQUAL(err, INFERO_ERROR_GENERAL_EXCEPTION);    

    EXPECT_EQUAL(infero_finalise(), INFERO_SUCCESS);
     
}


int main(int argc, char* argv[]) {
    return run_tests(argc, argv);
}