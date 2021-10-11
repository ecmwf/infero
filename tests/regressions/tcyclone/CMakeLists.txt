
if (HAVE_ONNX)

    # C++ runner
    ecbuild_configure_file(tcyclone_onnx.sh.in
        tcyclone_onnx.sh @ONLY )

    ecbuild_add_test(
        TYPE     SCRIPT
        COMMAND  tcyclone_onnx.sh )

    # F90 runner
    ecbuild_configure_file(f90_tcyclone_onnx.sh.in
        f90_tcyclone_onnx.sh @ONLY )

    ecbuild_add_test(
        TYPE     SCRIPT
        COMMAND  f90_tcyclone_onnx.sh )

endif()


if (HAVE_TF_LITE)

    # C++ runner
    ecbuild_configure_file(tcyclone_tflite.sh.in
        tcyclone_tflite.sh @ONLY )

    ecbuild_add_test(
        TYPE     SCRIPT
        COMMAND  tcyclone_tflite.sh )

    # C-runner
    ecbuild_configure_file(tcyclone_tflite_c.sh.in
        tcyclone_tflite_c.sh @ONLY )

    ecbuild_add_test(
        TYPE     SCRIPT
        COMMAND  tcyclone_tflite_c.sh )
endif()


if (HAVE_TENSORRT)

    # C++ runner
    ecbuild_configure_file(tcyclone_trt.sh.in
        tcyclone_trt.sh @ONLY )

    ecbuild_add_test(
        TYPE     SCRIPT
        COMMAND  tcyclone_trt.sh )
endif()
