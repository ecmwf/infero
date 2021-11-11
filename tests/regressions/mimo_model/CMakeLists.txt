
if (HAVE_ONNX)

    # C++ runner
    ecbuild_configure_file(mimo_onnx.sh.in
        mimo_onnx.sh @ONLY )

    ecbuild_add_test(
        TYPE     SCRIPT
        COMMAND  mimo_onnx.sh )

endif()


if (HAVE_TF_C)

    ecbuild_configure_file(mimo_tf_c.sh.in
        mimo_tf_c.sh @ONLY )

    ecbuild_add_test(
        TYPE     SCRIPT
        COMMAND  mimo_tf_c.sh )

endif()


if (HAVE_TF_LITE)

    ecbuild_configure_file(mimo_tflite.sh.in
        mimo_tflite.sh @ONLY )

    ecbuild_add_test(
        TYPE     SCRIPT
        COMMAND  mimo_tflite.sh )

endif()
