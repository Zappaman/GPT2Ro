# Rust text processing

## Description
- This section of the project currently includes the following utilities:
    -   `wc_filter` -> filter out lines that have too few words in them.
    -   `wc_histogram` -> generate a histogram on disk to see the distribution of word counts per line for your text dataset.

## Setup
- Just make sure you have rust [installed on your system](https://www.rust-lang.org/tools/install) 

## Usage
- `wc_filter`
```
cargo run --release --bin wc_filter -- -i <input_path.txt> -o <output_path.txt> -w <minimum_number_of_words>
```
- `wc_histogram`
```
cargo run --release --bin wc_histogram -- -i <input_path.txt> -o <output_histogram_path.png>
```
