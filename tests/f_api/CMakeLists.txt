# (C) Copyright 1996- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

# list( APPEND _fapi_tests tensor_set)
list( APPEND _fapi_tests)

# ONNX needed
if (HAVE_ONNX)
    list( APPEND _fapi_tests infero_model)
endif()


foreach( _test ${_fapi_tests} )

  ecbuild_configure_file(test_fapi_${_test}.F90.in
                         test_fapi_${_test}.F90 @ONLY )

  ecbuild_add_test(
      TARGET       infero_test_fapi_${_test}
      SOURCES      test_fapi_${_test}.F90
      ENVIRONMENT  ${test_environment}
      CONDITION    HAVE_FORTRAN
      LIBS         inferof )

endforeach()
