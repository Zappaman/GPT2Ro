# GPT2Ro

Fine-tuning GPT2 for Romanian with HuggingFace Transformers

## Repo presentation

### Motivation

- Currently support for fine-tuning of large GPT2-based models for lower-resource languages such as Romanian is rather scarcely documented - so I wanted to see hands-on how feasible it is to use the Huggingface Transformers library to obtain a decent model with limited training data and hardware capacity.

### Features:

- Pytorch implementation making heavy use of [Huggingface Transformers](https://github.com/huggingface/transformers) APIs
- Romanian dataset preparation scripts
  - For detailed instructions on how to prepare your dataset, consult the `README.md` included in the `data/` subfolder.
  - Most of the processing scripts were taken from the [Romanian-Transformers](https://github.com/dumitrescustefan/Romanian-Transformers) repository - however at the time I looked through them I needed to perform some modifications in order to get them running correctly. Please check out their work as well if you're interested in the subject!
- Tensorboard logging during training
- [Yacs](https://github.com/rbgirshick/yacs) based experiment configurations
- Fp16/fp32 training modes
- Command-line generation sentences with various strategies as described [here](https://huggingface.co/blog/how-to-generate): greedy, beam search, top-k, top-p, top-k/top-p
- Rust inference for deployment using the [rust-bert](https://github.com/guillaume-be/rust-bert) library

## Repo Setup

- Ubuntu >= 16.04, strong GPU recommended.
- Python >= version 3.7
- Using an [Anaconda](https://www.anaconda.com/) environment is encouraged.
- Install [Pytorch](https://pytorch.org/)
- Run `pip install -r ./requirements.txt`

## Usage

### Dataset setup

- Prepare your dataset as indicated in `data/README.md`
- Create a romanian tokenizer for your new dataset.
  ```
  python create_tokenizer.py --corpus_path=<path_to_corpus> --ref_tokenizer=<type_of_tokenizer> --output_path=<path where tokenizer files are placed>
  ```

### Training

- Run
  ```
  python train.py --cfg=experiments/gpt2_ro/unfrozen.yml
  ```
  - Check out the included yaml configurations in the `experiments` subfolder to get an idea of how to parameterize the training procedure.

### Testing

- Run
  ```
  python generate.py --model_path=<your_model_path> --tokenizer_prefix_path=<your_tokenizer_prefix_path>
  ```

### Model deployment

- I've also done some preliminary work in seeing how I can use a Pytorch model in Rust with the help of the [rust-bert](https://github.com/guillaume-be/rust-bert) framework to get better performance better suited for a production-ready environment.
- Currently I've provided support for:
  - Converting the Pytorch model to a format accepted by `rust-bert`
  - Running the model in a small command-line app -> either by simulating a dialogue or simply issuing requests to the model in a loop.
- The use-cases can definitely be extended, such as providing an online text generation service for GPT2, but I've considered that to be out-of-scope for this project.
- For more detailed information on GPT2 deployment in Rust, consult `model_deployment/README.md`

## Current results

- Intelligible Romanian can be obtained after 7-8 hours of training with the `./experiments/gpt2_ro/unfrozen.yml` config. Some selected results generated via `top-k top-p sampling` are below:

```
Example 1.
Input: "Ne trebuie ceva"
Output: "Ne trebuie ceva bun. Trebuie doar ca oamenii să nu fi auzit în mod corect. Ce e mai firesc și că ai un plan, și să nu-i lași pe cei care i-au trimis."

Example 2.
Input: "Am auzit"
Output: "Am auzit asta de fiecare dată. Mi-am imaginat că au început să creadă că am fost pe o cale să citesc cartea. Am luat ceva în comun în timp ce am văzut că este foarte bine să am o temă de lectură."

Example 3.
Input: "Ce anume"
Output: "Ce anume ești tu? Este în prezent, în loc să faci o schimbare de mentalitate. Și pentru că am fost prea fericită pentru că, de exemplu, oamenii sunt în acest lucru, trebuie să nu îi pasă de ce nu ne uităm."
```

- Shortly after reaching this performance the training seems to diverge: the model either refuses to generate anything beyond the input or continuously repeats the same words.
- Currently I haven't seen any improvement using gradual unfreezing as opposed to training using all layers from the start.
- My fine-tuning experiments were done based on the default 'gpt2' English model and weights made available by the Transformers library. It would be interesting to see the performance of other larger gpt2 models, such as the medium or large ones, although this would require more computational resources.
