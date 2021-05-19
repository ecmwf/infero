
if (HAVE_ONNX)

    ecbuild_configure_file(tcyclone_onnx.sh.in
        tcyclone_onnx.sh @ONLY )

    ecbuild_add_test(
        TYPE     SCRIPT
        COMMAND  tcyclone_onnx.sh )
endif()

if (HAVE_TF_LITE)

    ecbuild_configure_file(tcyclone_tflite.sh.in
        tcyclone_tflite.sh @ONLY )

    ecbuild_add_test(
        TYPE     SCRIPT
        COMMAND  tcyclone_tflite.sh )
endif()
