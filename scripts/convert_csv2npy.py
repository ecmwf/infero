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
import argparse
from load_data import load_data

if __name__ == "__main__":
    """
    Lightweight script to convert a CSV data into numpy format
    """
    
    parser = argparse.ArgumentParser("Data Augmentation")
    parser.add_argument('input', help="Path of the input CSV")
    parser.add_argument('output', help="Path of the output npy file")
    parser.add_argument('--channels',
                        help="NB of batch-elements to write to output",
                        default=1,
                        type=int)
    
    args = parser.parse_args()
    
    # it assumes that batch size is the first
    npy_data = load_data(args.input)
    #npy_data = npy_data.reshape((1, -1))
    
    out_data = npy_data[:args.channels, :]
    
    print(f"Writing numpy of shape {out_data.shape} into {args.output}")
    print(f"Output numpy {out_data}")
    
    np.save(args.output, out_data.astype(np.float32))
