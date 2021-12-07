#include <stdlib.h>
#include <stdio.h>
#include "infero/api/infero.h"
#include "eckit/testing/Test.h"

using namespace eckit::testing;


CASE("infero_init"){

    char* arg1 = (char*)"arg1";
    char** args;
    args = &arg1;
    int argc = 1;
    
    // initialise infero lib
    EXPECT_EQUAL(infero_initialise(argc, args), INFERO_SUCCESS);

    // error: already initialised!
    EXPECT_EQUAL(infero_initialise(argc, args), INFERO_ERROR_GENERAL_EXCEPTION);

    // finalise
    EXPECT_EQUAL(infero_finalise(), INFERO_SUCCESS);

    // error: already finalised!
    EXPECT_EQUAL(infero_finalise(), INFERO_ERROR_GENERAL_EXCEPTION);
}


int main(int argc, char* argv[]) {
    return run_tests(argc, argv);
}