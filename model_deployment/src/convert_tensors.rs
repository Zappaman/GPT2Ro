// Modified from https://github.com/guillaume-be/rust-bert/blob/master/src/convert-tensor.rs

extern crate anyhow;
extern crate tch;
use structopt::StructOpt;

#[derive(StructOpt, Debug)]
#[structopt(name = "converter")]
struct Opt {
    #[structopt(short, long)]
    python_tensors_path: String,
    #[structopt(short, long)]
    rust_tensors_path: String,
}

pub fn main() -> anyhow::Result<()> {
    let opt = Opt::from_args();

    let tensors = tch::Tensor::read_npz(opt.python_tensors_path)?;
    tch::Tensor::save_multi(&tensors, opt.rust_tensors_path)?;

    Ok(())
}
