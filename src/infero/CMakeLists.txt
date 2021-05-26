add_subdirectory(api)
add_subdirectory(clustering)
add_subdirectory(ml_engines)


# infero.x executable
ecbuild_add_executable( TARGET infero.x
    SOURCES   infero.cc
    CONDITION HAVE_BUILD_TOOLS
    INCLUDES  ${eckit_INCLUDE_DIRS}
    LIBS      infero eckit eckit_option )

