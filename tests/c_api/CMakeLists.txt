# (C) Copyright 1996- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


ecbuild_add_test(TARGET        infero_test_infero_init_c
                 INCLUDES      ${eckit_INCLUDE_DIRS}
                 SOURCES       test_infero_init.cc
                 LIBS          inferoapi eckit
)

if (HAVE_TF_C)

    # test c-api with tf_c inference model
    ecbuild_configure_file(test_model_tfc_c_api.cc.in
                           test_model_tfc_c_api.cc @ONLY )

    ecbuild_add_test(TARGET       infero_test_model_tfc_c_api
                    INCLUDES      ${eckit_INCLUDE_DIRS}
                    SOURCES       test_model_tfc_c_api.cc
                    LIBS          inferoapi eckit
    )

endif()

