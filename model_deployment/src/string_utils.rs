extern crate anyhow;

pub fn get_uptolast(input_str: &str, sep: &str, add_term: bool) -> anyhow::Result<String> {
    let mut output_split: Vec<&str> = input_str.split(sep).collect();
    output_split.truncate(output_split.len() - 1);
    let mut output_str = output_split.join(sep);
    if add_term {
        output_str.push_str(sep);
    }
    Ok(output_str)
}
