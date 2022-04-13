
list( APPEND _fapi_tests tensor_set)

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
