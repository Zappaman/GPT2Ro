from pathlib import Path
from tokenizers import ByteLevelBPETokenizer
from transformers import GPT2Tokenizer
import argparse
import os


def train_tokenizer_ro(data_path: str, ref_tokenizer_str: str, output_dir: str) -> None:
    ref_tokenizer = GPT2Tokenizer.from_pretrained(ref_tokenizer_str)

    # Create GPT2 tokenizer with same size as GPT2 from which we want to fine-tune
    vocab_size_pretrained = ref_tokenizer.vocab_size
    paths = [data_path]
    tokenizer = ByteLevelBPETokenizer()

    tokenizer.train(files=paths, vocab_size=vocab_size_pretrained, min_frequency=2, special_tokens=[
        "<pad>",
        "<|endoftext|>"
    ])

    # Save files to disk
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    tokenizer.save_model(output_dir, "gpt2_tokenizer_ro")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Romanian tokenizer utility script")
    parser.add_argument("--corpus_path", type=str,
                        help="Path to your romanian corpus.txt")
    parser.add_argument("--ref_tokenizer", type=str,
                        help="Type of huggingface GPT2 tokenizer to reference in creating the vocab size")
    parser.add_argument("--output_path", type=str,
                        help="Path to place the output tokenizer")
    args = parser.parse_args()

    train_tokenizer_ro(args.corpus_path, args.ref_tokenizer, args.output_path)
