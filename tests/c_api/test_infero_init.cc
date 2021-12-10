/*
 * (C) Copyright 1996- ECMWF.
 *
 * This software is licensed under the terms of the Apache Licence Version 2.0
 * which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
 * In applying this licence, ECMWF does not waive the privileges and immunities
 * granted to it by virtue of its status as an intergovernmental organisation
 * nor does it submit to any jurisdiction.
 */

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