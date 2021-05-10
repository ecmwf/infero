import argparse
from load_data import load_data


if __name__ == "__main__":
    """
    Lightweight script to convert a CSV data into numpy format
    """
    
    parser = argparse.ArgumentParser("Data Augmentation")
    parser.add_argument('reference', help="Path of the reference data (CSV or NPY)")
    parser.add_argument('prediction', help="Path of the reference data (CSV or NPY)")
    parser.add_argument('--channels',
                        help="NB of batch-elements to write to output",
                        default=-1,
                        type=int)
    
    args = parser.parse_args()

    ref_data = load_data(args.reference)[:]
    print(f"ref_data.shape {ref_data}")
    # ref_data = load_data(args.reference)
    # print(f"ref_data.shape {ref_data.shape}")
    
    pred_data = load_data(args.prediction)[:]
    print(f"pred_data.shape {pred_data}")
    # pred_data = load_data(args.prediction)
    # print(f"pred_data.shape {pred_data.shape}")
    
    # diff_data = (ref_data-pred_data)
