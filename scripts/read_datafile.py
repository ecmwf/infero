
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

