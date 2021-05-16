# GPT2 Deployment in Rust

## Motivation
- Once we are done prototyping and training our Pytorch models in Python and we're ready to integrate our models in a real world application, it is time to start optimizing for speed. 
- Although a lot of optimizations for custom logic is available in Python depending on the given scenario, it is generally preferable to decouple the model from Python and run it in a faster language such as C++ - or in this repo's case, Rust.

## Setup
- Make sure you have rust [installed on your system](https://www.rust-lang.org/tools/install) 
- I recommend following the instructions from [rust-bert](https://github.com/guillaume-be/rust-bert#manual-installation-recommended) for setting up [libtorch](https://pytorch.org/cppdocs/installing.html) on your system in order to benefit from the full speed of CUDA if you plan on running this on a system with GPU. Simply running the examples here without this setup means you will be using a cpu-only libtorch downloaded by default with the `rust-bert` project dependency when building the rust executable.

## Usage
- First, convert the Pytorch weights to a Rust-friendly format:
    ```
    python python_utils/convert_model.py --model_path <path_to_pytorch_model.ptr> --output_path <folder_to_output_rust_weights>
    ```
- Then build and run the gpt2 executable via:
    ```
    cargo run --bin gpt2 --release -- -m <path_to_rust_exported_model_weights_file> -t <path_to_tokenizer_folder> -o <loop|dialog>
    ```
    - `-o loop` will simply run the gpt2 inference in a loop, discarding previous runs, while `-o dialog` will prepend the conversation you're having as context for future gpt2 runs.
- I used the great [rust-bert](https://github.com/guillaume-be/rust-bert) library to run the Pytorch trained model in Rust.
