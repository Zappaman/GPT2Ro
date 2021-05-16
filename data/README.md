# Data processing for Fine-Tuning GPT2 in Romanian
- Much of this repository is based on the corpus creation of the [Romanian-Transformers](https://github.com/dumitrescustefan/Romanian-Transformers/) repo. The majority of the `README` instructions here are taken directly from this repo.
- `Note` The instructions from the above repo failed in the case of OPUS, hence some modifications were made to properly add that part of the corpus as well. 
# Corpus componentes 
- [Romanian Wikipedia](https://dumps.wikimedia.org/rowiki/) 
- [OSCAR](https://oscar-corpus.com/) 
- Romanian [OPUS](https://opus.nlpl.eu/) monolingual data

# Corpus preparation instructions:
## First install the dependencies:
```
pip install -r requirements.txt
```
- Run the following commands:
```shell script
./wiki_download.sh
./opus_download.sh
./oscar_download.sh
```
The scrips will create the ``raw`` folder and download the corpora there.

## Clean the corpus by running:
```shell script
python3 wiki_clean.py
python3 opus_clean.py
python3 oscar_clean.py
``` 
The scripts will take a few hours to clean the corpora. The folder ``clean`` will be created with the txt files for each corpus. 

## Merge the corpora
The script will concatenate the corpora while extracting a validation subset of sentences that respects the distribution of line in each composing corpus.
```shell script
python merge_corpora.py
```
The folder ``merged`` will now contain a ``train.txt`` and a ``valid.txt``.
Edit the ``valid_count = 5000`` in ``merge_corpora.py`` to change the default number of lines that will be extracted for validation.


## Extra rust processing (Currently optional)
- I have also played around with some text preprocessing in Rust, which can be found under `data/processing/rust_processing`.
- Rust programs are clearly faster (similar to C++) than pure python (3-5x times faster in worst case scenario, much faster when more custom logic is involved) and thanks to the `cargo` package management system I find it much easier to create an actual working setup with various dependencies than C++ would allow.
- This part is not actually needed to create the data necessary for training the model, I just included it as a possible baseline for future fast processing utilities.
