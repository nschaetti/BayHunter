
# Imports
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import glob
import re


# Plot likelihoods
def plot_likelihoods(directory):
    """
    Plot likelihoods from all chains in the given directory.
    """
    pattern = os.path.join(directory, "c???_p2likes.npy")
    files = sorted(glob.glob(pattern))

    if not files:
        print(f"No files matching 'c*_p2likes.npy' found in {directory}")
        return
    # end if

    # Plot
    plt.figure(figsize=(10, 6))

    # Plot each chain
    for f in files:
        data = np.load(f)
        label = os.path.basename(f).replace("_p1likes.npy", "")
        plt.plot(data, label=label, alpha=0.8)
    # end for

    # Labels
    plt.xlabel("Iteration")
    plt.ylabel("Log-Likelihood")
    plt.title("Evolution of Log-Likelihood per Chain")
    plt.legend(loc="best", fontsize="small")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
# end plot_likelihoods


def plot_likelihood_curves(directory, tail=500):
    pattern = os.path.join(directory, "c???_p1likes.npy")
    files = sorted(glob.glob(pattern))

    if not files:
        print("No likelihood files found.")
        return

    plt.figure(figsize=(10, 5))

    for f in files:
        log_likelihood = np.load(f)

        # Take the last `tail` entries
        log_tail = log_likelihood[-tail:]

        # Convert to likelihood
        likelihood = np.exp(log_tail - np.max(log_tail))  # for numerical stability

        label = os.path.basename(f).split("_")[0]
        plt.plot(likelihood, label=label, alpha=0.7)

    plt.title("Likelihood evolution over last iterations")
    plt.xlabel("Iterations (last {})".format(tail))
    plt.ylabel("Likelihood (normalized)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
# end plot_likelihood_curves


# Main function
def main():
    parser = argparse.ArgumentParser(description="Plot BayHunter p1likes for all chains")
    parser.add_argument("directory", type=str, help="Directory containing c*_p1likes.npy files")
    parser.add_argument("--tail", type=int, default=500, help="Number of last iterations to plot")
    args = parser.parse_args()
    plot_likelihoods(args.directory)
    # plot_likelihood_curves(args.directory, args.tail)
# end main


if __name__ == "__main__":
    main()
# end if
