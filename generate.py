"""
Different kinds of generating text with a pretrained gpt2 model
Strategies taken from https://huggingface.co/blog/how-to-generate
"""


def generate_beams(model, tokenizer, text, max_length):
    """
    Use beam search
    """
    input_ids = tokenizer.encode(text, return_tensors='pt')
    beam_output = model.generate(
        input_ids, max_length=50, num_beams=5, early_stopping=True)
    return tokenizer.decode(beam_output[0], skip_special_tokens=True)


def generate_topk(model, tokenizer, text, max_length):
    """
    User top-k sampling
    """
    input_ids = tokenizer.encode(text, return_tensors='pt')
    sample_output = model.generate(
        input_ids, do_sample=True, max_length=50, top_k=50)
    return tokenizer.decode(sample_output[0], skip_special_tokens=True)


def generate_topp(model, tokenizer, text, max_length):
    """
    User top-p sampling
    """
    input_ids = tokenizer.encode(text, return_tensors='pt')
    sample_output = model.generate(
        input_ids, do_sample=True, max_length=50, top_k=0, top_p=0.92)
    return tokenizer.decode(sample_output[0], skip_special_tokens=True)


def generate_topktopp(model, tokenizer, text, max_length):
    """
    User top-k top-p sampling
    """
    input_ids = tokenizer.encode(text, return_tensors='pt').cuda()
    sample_output = model.generate(
        input_ids, do_sample=True, max_length=50, top_k=50, top_p=0.95)
    return tokenizer.decode(sample_output[0], skip_special_tokens=True)


def test_generation(model, tokenizer, max_length=50):
    s_list = ['Ne trebuie', 'Ce frumos', "Ai auzit"]

    out_list = [generate_topktopp(
        model, tokenizer, s, max_length) for s in s_list]
    return out_list


if __name__ == "__main__":
    import argparse
    import torch
    from transformers import GPT2Tokenizer

    parser = argparse.ArgumentParser(
        "Generate text using a fine-tuned GPT2 model and given tokenizer")
    parser.add_argument("--model_path", type=str,
                        help="Path to your saved model")
    parser.add_argument("--tokenizer_prefix_path", type=str,
                        help="The common path to your merges.txt and vocab.json files of your tokenizer")

    args = parser.parse_args()
    tokenizer_path_prefix = args.tokenizer_prefix_path

    model = torch.load(args.model_path)['model']
    if torch.cuda.is_available():
        model = model.cuda()

    merge_txt = f"{tokenizer_path_prefix}merges.txt"
    vocab_json = f"{tokenizer_path_prefix}vocab.json"

    tokenizer = GPT2Tokenizer(
        vocab_json,
        merge_txt,
        pad_token="<pad>",
    )
    print("Type in your input for the model:")
    sentence = input()
    print(generate_topktopp(model, tokenizer, sentence, max_length=100))
