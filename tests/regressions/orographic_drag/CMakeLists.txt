
if (HAVE_ONNX)

    ecbuild_configure_file(orographic_drag_onnx.sh.in
        orographic_drag_onnx.sh @ONLY )

    ecbuild_add_test(
        TYPE     SCRIPT
        COMMAND  orographic_drag_onnx.sh )
endif()

if (HAVE_TF_LITE)

    ecbuild_configure_file(orographic_drag_tflite.sh.in
        orographic_drag_tflite.sh @ONLY )

    ecbuild_add_test(
        TYPE     SCRIPT
        COMMAND  orographic_drag_tflite.sh )
endif()
