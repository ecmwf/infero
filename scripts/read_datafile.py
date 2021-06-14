/*
 * (C) Copyright 1996- ECMWF.
 *
 * This software is licensed under the terms of the Apache Licence Version 2.0
 * which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
 * In applying this licence, ECMWF does not waive the privileges and immunities
 * granted to it by virtue of its status as an intergovernmental organisation
 * nor does it submit to any jurisdiction.
 */

import argparse
import numpy as np
from load_data import load_data
import matplotlib.pyplot as plt

if __name__ == "__main__":
    """
    Lightweight script to plot prediction of shape [batch, xdim, ydim, channel]
    """
    
    parser = argparse.ArgumentParser("Data Reader")
    parser.add_argument('data_file', help="Path of the data (CSV or NPY)")
    parser.add_argument('channel',
                        help="channel ID",
                        default=-1,
                        type=int)

    args = parser.parse_args()

    print(f"Selected: data_file {args.data_file}, channel {args.channel}")
    
    # read data like:
    # - batch size
    # - img x
    # - img y
    # - channel
    vv = load_data(args.data_file)[:, :, :, args.channel]
    assert len(vv.shape) == 3

    print(f"shape {vv.shape}")
    print(f"max {np.max(vv)}")
    print(f"min {np.min(vv)}")
    print(f"mean {np.mean(vv)}")
    
    print(f"\n-----------------")
    print(f"press q to exit..")
    print(f"-----------------")
    
    plt.imshow(vv[0, :, :])
    plt.show()

