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
import numpy as np


def load_data(_data_path, delimiter=' '):
    """
    Load CSV or numpy data
    :param _data_path:
    :param delimiter:
    :return:
    """
    
    if not os.path.exists(_data_path):
        return None
    
    _ext = os.path.splitext(_data_path)[1]
    
    if _ext == ".csv":
        return np.genfromtxt(_data_path, delimiter=delimiter)
    elif _ext == ".npy":
        return np.load(_data_path)
    else:
        print(f"Extension {_ext} not supported!")
        return None
