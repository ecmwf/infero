
list( APPEND _fapi_tests
  infero_model
  tensor_set
)

foreach( _test ${_fapi_tests} )

ecbuild_add_test(
    TARGET       test_fapi_${_test}
    SOURCES      test_fapi_${_test}.F90
    ENVIRONMENT  ${test_environment}
    CONDITION    HAVE_FORTRAN
    LIBS         inferof )
endforeach()