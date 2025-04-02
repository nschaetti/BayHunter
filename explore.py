
# Imports
import argparse
import numpy as np
import os


def main():
    """
    Main function
    """
    parser = argparse.ArgumentParser(description="Inspect a .npy file")
    parser.add_argument("file", type=str, help="Path to the .npy file")
    parser.add_argument("--summary", action="store_true", help="Only show summary info (shape, dtype, etc.)")
    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"File not found: {args.file}")
        return
    # end if

    try:
        data = np.load(args.file, allow_pickle=True)

        print(f"Loaded: {args.file}")
        print(f"Type: {type(data)}")

        if isinstance(data, np.ndarray):
            print(f"Shape: {data.shape}")
            print(f"Dtype: {data.dtype}")
            if not args.summary:
                print("Contents:")
                print(data)
        else:
            print("Non-ndarray object (likely list/dict from pickle):")
            if not args.summary:
                print(data)

    except Exception as e:
        print(f"Error loading file: {e}")
    # end try
# end main

if __name__ == "__main__":
    main()
# end if
