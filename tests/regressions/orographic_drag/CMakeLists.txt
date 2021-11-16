
if (HAVE_ONNX)

    ecbuild_configure_file(orographic_drag_onnx.sh.in
        orographic_drag_onnx.sh @ONLY )

    ecbuild_add_test(
        TYPE     SCRIPT
        COMMAND  orographic_drag_onnx.sh )

    # F90 runner
    ecbuild_configure_file(f90_oro_drag_onnx.sh.in
        f90_oro_drag_onnx.sh @ONLY )

    ecbuild_add_test(
        TYPE     SCRIPT
        COMMAND  f90_oro_drag_onnx.sh )

endif()

if (HAVE_TF_C)

    ecbuild_configure_file(orographic_drag_tf_c.sh.in
        orographic_drag_tf_c.sh @ONLY )

    ecbuild_add_test(
        TYPE     SCRIPT
        COMMAND  orographic_drag_tf_c.sh )

endif()


if (HAVE_TF_LITE)

    ecbuild_configure_file(orographic_drag_tflite.sh.in
        orographic_drag_tflite.sh @ONLY )

    ecbuild_add_test(
        TYPE     SCRIPT
        COMMAND  orographic_drag_tflite.sh )
endif()

if (HAVE_TENSORRT)

    ecbuild_configure_file(orographic_drag_trt.sh.in
        orographic_drag_trt.sh @ONLY )

    ecbuild_add_test(
        TYPE     SCRIPT
        COMMAND  orographic_drag_trt.sh )
endif()
