#!/usr/bin/env python3
import argparse
import numpy as np
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description="Plot error data from saved .npz file")
    parser.add_argument("record_name", type=str, help="Name of the record to plot")
    args = parser.parse_args()

    # Load data
    # filepath = f"test_run_new_3/error_{args.record_name}.npz"
    data = np.load(args.record_name, allow_pickle=True)

    # Plot
    plt.figure()
    plt.plot(data["long_error"])
    plt.xlabel(str(data["xlabel"]))
    plt.ylabel(str(data["ylabel"]))
    plt.title(str(data["title"]))
    plt.show()

if __name__ == "__main__":
    main()
