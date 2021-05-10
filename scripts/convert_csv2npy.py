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

    npy_data = load_data(args.input)[:args.channels]
    
    out_data = npy_data[:args.channels]
    
    print(f"Writing numpy of shape {out_data.shape} into {args.output}")
    print(f"Output numpy {out_data}")
    
    np.save(args.output, out_data.astype(np.float32))
