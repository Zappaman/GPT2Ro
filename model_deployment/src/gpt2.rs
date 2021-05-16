extern crate anyhow;

use structopt::StructOpt;

mod generation_utils;
mod string_utils;

#[derive(StructOpt, Debug)]
#[structopt(name = "generator")]
struct Opt {
    #[structopt(short, long)]
    model: String,
    #[structopt(short, long)]
    token: String,
    #[structopt(short = "o", long)]
    mode: String,
}

fn main() -> anyhow::Result<()> {
    let opt = Opt::from_args();

    let tok_path = opt.token.as_str();
    let model_path = opt.model.as_str();
    let mode = opt.mode.as_str();
    match mode {
        "dialog" => generation_utils::gpt2_dialog(tok_path, model_path),
        "loop" => generation_utils::gpt2_loop(tok_path, model_path),
        _ => Ok(()),
    }
}
