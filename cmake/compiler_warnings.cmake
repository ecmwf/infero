ecbuild_add_option( FEATURE WARNINGS
                    DEFAULT ON
                    DESCRIPTION "Add warnings to compiler" )

# activate warnings, ecbuild macros check the compiler recognises the options
if(HAVE_WARNINGS) 

  ecbuild_add_cxx_flags("-Wall")
  ecbuild_add_cxx_flags("-Wextra")

  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    ecbuild_add_cxx_flags("-Wno-unused-parameter")
    ecbuild_add_cxx_flags("-Wno-unused-variable")
    ecbuild_add_cxx_flags("-Wno-sign-compare")
  endif()

  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    ecbuild_add_cxx_flags("-Wno-unused-parameter")
    ecbuild_add_cxx_flags("-Wno-unused-variable")
    ecbuild_add_cxx_flags("-Wno-sign-compare")
  endif()

  endif()

