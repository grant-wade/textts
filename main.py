import os
import re
import sys


def split_book_to_pages(input_path):
    # Create output directory based on input filename
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    output_dir = f"{base_name}_pages"
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Open the input file for reading
    with open(input_path, "r", encoding="utf-8") as infile:
        lines = infile.readlines()

    current_page = None
    current_file = None
    page_pattern = re.compile(r"^\d+\s*$")

    for line in lines:
        if page_pattern.match(line):
            # Close the current file if it exists
            if current_file is not None:
                current_file.close()
            # Increment page number
            current_page = 0 if current_page is None else current_page + 1
            # Create new filename with leading zeros to maintain order
            filename = os.path.join(output_dir, f"page_{current_page:03d}.txt")
            current_file = open(filename, "w", encoding="utf-8")
        else:
            # If we haven't found the first page marker yet, create page_000
            if current_page is None:
                current_page = 0
                filename = os.path.join(output_dir, f"page_{current_page:03d}.txt")
                current_file = open(filename, "w", encoding="utf-8")
        else:
            if current_file is not None:
                current_file.write(line)

    # Close the last file if it's open
    if current_file is not None:
        current_file.close()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python main.py <input_file>")
        sys.exit(1)
    input_file = sys.argv[1]
    split_book_to_pages(input_file)
