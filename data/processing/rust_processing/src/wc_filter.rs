use anyhow::Result;
use std::fs::OpenOptions;
use std::io::prelude::*;
use std::io::{self};
use structopt::StructOpt;

mod utils;

#[derive(StructOpt, Debug)]
#[structopt(name = "processing")]
struct Opt {
    #[structopt(short, long)]
    input_file: String,

    #[structopt(short, long)]
    output_file: String,

    #[structopt(short, long)]
    words: u64,
}

fn main() -> Result<()> {
    let opt = Opt::from_args();
    let minimum_word_count = opt.words;

    if let Ok(lines) = utils::file_utils::read_lines(opt.input_file.as_str()) {
        let mut output_file = io::BufWriter::new(
            OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(true)
                .open(opt.output_file)
                .unwrap(),
        );

        for line in lines {
            let line = line?;

            let mut wc = 0;
            for _ in line.split_whitespace() {
                wc += 1;
            }
            if wc < minimum_word_count as usize {
                continue; // skipping line in filtered set
            }
            output_file.write((line + "\n").as_bytes())?;
        }
    }

    Ok(())
}
