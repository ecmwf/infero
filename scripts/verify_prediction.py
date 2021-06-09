import numpy as np
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from load_data import load_data


if __name__ == "__main__":
    """
    Lightweight script to convert a CSV data into numpy format
    """
    
    parser = argparse.ArgumentParser("Data Augmentation")
    parser.add_argument('reference', help="Path of the reference data (CSV or NPY)")
    parser.add_argument('prediction', help="Path of the reference data (CSV or NPY)")
    parser.add_argument('--channel',
                        help="NB of batch-elements to write to output",
                        default=-1,
                        type=int)
    parser.add_argument('--save_img_path', help="Path to output img")
    
    args = parser.parse_args()

    if args.channel is not None:
        ref_data = load_data(args.reference)[args.channel]
        pred_data = load_data(args.prediction)[args.channel]
    else:
        ref_data = load_data(args.reference)
        pred_data = load_data(args.prediction)

    print(f"ref_data.shape {ref_data.shape}")
    print(f"pred_data.shape {pred_data.shape}")
    
    rel_diff_data_percent = (ref_data-pred_data)/ref_data

    print(f"Rel diff avg {np.mean(rel_diff_data_percent):20.10f} [%]")
    print(f"Rel diff max {np.max(rel_diff_data_percent):20.10f} [%]")
    print(f"Rel diff min {np.min(rel_diff_data_percent):20.10f} [%]")

    if args.save_img_path:
        fig, ax = plt.subplots()
        im = plt.imshow(rel_diff_data_percent[:, :, 0])
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        plt.savefig(args.save_img_path)
        plt.show()



