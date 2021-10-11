#
# (C) Copyright 1996- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

import os
import copy
import cffi
import platform
import numpy as np


class Infero:
    """
    Minimal class that wraps the infero C API
    """

    def __init__(self, model_path, model_type):

        # path to infero model
        self.model_path = model_path

        # model type (see available infero backends)
        self.model_type = model_type

        # inference configuration string
        self.config_str = f"path: {self.model_path}\ntype: {self.model_type}"

        # ffi lib
        self.ffi = None
        self.__lib = None

        # C API handle
        self.infero_hdl = None

    def initialise(self):
        """
        Initialise the library
        :return:
        """

        # ffi lib
        self.ffi = cffi.FFI()

        self.ffi.cdef(self.__read_header(), override=True)
        libName = {
            'Linux': 'libinferoapi.so'
        }

        self.__lib = self.ffi.dlopen(libName[platform.system()])

        # main args not directly used by the API
        args = [""]
        cargs = [self.ffi.new("char[]", ar.encode('ascii')) for ar in args]
        argv = self.ffi.new(f'char*[]', cargs)

        # init infero lib
        self.__lib.infero_initialise(len(cargs), argv)
        config_cstr = self.ffi.new("char[]", self.config_str.encode('ascii'))

        # get infero handle
        self.infero_hdl = self.__lib.infero_create_handle_from_yaml_str(config_cstr)

        # open the handle
        self.__lib.infero_open_handle(self.infero_hdl)

    def infer(self, input_data, output_shape):
        """
        Run Inference
        :param input_data:
        :param output_shape:
        :return:
        """

        # input set to Fortran order
        input_data = np.array(input_data, order='C', dtype=np.float32)
        cdata1p = self.ffi.cast("float *", input_data.ctypes.data)
        cshape1 = self.ffi.new(f"int[]", input_data.shape)

        # output also expected in Fortran order
        cdata2 = np.zeros(output_shape, order='C', dtype=np.float32)
        cdata2p = self.ffi.cast("float *", cdata2.ctypes.data)
        cshape2 = self.ffi.new(f"int[]", output_shape)

        self.__lib.infero_inference_float_ctensor(self.infero_hdl,
                                                  cdata1p, len(input_data.shape), cshape1,
                                                  cdata2p, len(output_shape), cshape2)

        return_output = copy.deepcopy(cdata2)
        return_output = np.array(return_output)

        return return_output

    def finalise(self):
        """
        Finalise the Infero API
        :return:
        """

        # close the handle
        self.__lib.infero_close_handle(self.infero_hdl)

        # delete the handle
        self.__lib.infero_delete_handle(self.infero_hdl)

        # finalise
        self.__lib.infero_finalise()

    def __read_header(self):
        with open(os.path.join(os.path.dirname(__file__), 'pyinfero-headers.h'), 'r') as f:
            return f.read()
