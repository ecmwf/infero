# infero-version executable
ecbuild_add_executable( TARGET infero-version
    SOURCES   infero_version.cc
    CONDITION HAVE_BUILD_TOOLS
    INCLUDES  infero_version.h
)

add_subdirectory(infero)
