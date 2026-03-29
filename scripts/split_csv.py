"""
split_csv.py
============
Split a CSV into smaller files with N rows each.

Features:
- Skips empty rows
- Preserves header in every file
- CLI-based

Usage:
    python split_csv.py --input data.csv --rows 10 --output_dir chunks
"""

import argparse
import csv
import os


def is_empty_row(row):
    return all((cell is None or str(cell).strip() == "") for cell in row.values())


def split_csv(input_file, rows_per_file, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    with open(input_file, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames

        chunk = []
        file_count = 1
        total_rows = 0

        for row in reader:
            if is_empty_row(row):
                continue  # skip empty rows

            chunk.append(row)
            total_rows += 1

            if len(chunk) == rows_per_file:
                write_chunk(chunk, headers, output_dir, file_count)
                file_count += 1
                chunk = []

        # write remaining rows
        if chunk:
            write_chunk(chunk, headers, output_dir, file_count)

    print(f"✅ Done! {file_count} file(s) created with {total_rows} valid rows.")


def write_chunk(chunk, headers, output_dir, index):
    output_file = os.path.join(output_dir, f"chunk_{index}.csv")

    with open(output_file, "w", newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(chunk)

    print(f"📄 Created: {output_file} ({len(chunk)} rows)")


def main():
    parser = argparse.ArgumentParser(description="Split CSV into smaller files")

    parser.add_argument("--input", "-i", required=True, help="Input CSV file")
    parser.add_argument("--rows", "-r", type=int, default=10, help="Rows per file (default: 10)")
    parser.add_argument("--output_dir", "-o", default="chunks", help="Output directory")

    args = parser.parse_args()

    split_csv(args.input, args.rows, args.output_dir)


if __name__ == "__main__":
    main()

'''
python split_csv.py --input data.csv --rows 10 --output_dir chunks
'''