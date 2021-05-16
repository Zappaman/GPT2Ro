extern crate anyhow;

use rust_bert::gpt2::Gpt2ConfigResources;
use rust_bert::pipelines::common::ModelType;
use rust_bert::pipelines::text_generation::{TextGenerationConfig, TextGenerationModel};
use rust_bert::resources::{LocalResource, RemoteResource, Resource};
use std::io::{self, Write};
use std::path::PathBuf;

use super::string_utils;

pub fn create_text_generation_model(
    tok_path: &str,
    model_path: &str,
) -> anyhow::Result<TextGenerationModel> {
    let vocab_path = format!("{}-vocab.json", tok_path);
    let merges_path = format!("{}-merges.txt", tok_path);

    let vocab_resource = Resource::Local(LocalResource {
        local_path: PathBuf::from(vocab_path),
    });
    let merges_resource = Resource::Local(LocalResource {
        local_path: PathBuf::from(merges_path),
    });
    let weights_resource = Resource::Local(LocalResource {
        local_path: PathBuf::from(model_path),
    });

    let generate_config = TextGenerationConfig {
        model_type: ModelType::GPT2,
        max_length: 100,
        do_sample: true,
        num_beams: 5,
        temperature: 1.1,
        num_return_sequences: 1,
        config_resource: Resource::Remote(RemoteResource::from_pretrained(
            Gpt2ConfigResources::GPT2,
        )),
        vocab_resource: vocab_resource,
        merges_resource: merges_resource,
        model_resource: weights_resource,
        ..Default::default()
    };
    let model = TextGenerationModel::new(generate_config)?;
    Ok(model)
}

pub fn gpt2_loop(tok_path: &str, model_path: &str) -> anyhow::Result<()> {
    let model = create_text_generation_model(tok_path, model_path)?;

    let mut done = false;
    while !done {
        print!("Input: ");
        io::stdout().flush()?;
        let mut read_buffer = String::new();
        io::stdin().read_line(&mut read_buffer)?;
        done = match read_buffer.as_str().trim() {
            "q" => true,
            _ => false,
        };
        if done {
            break;
        }
        println!("Generating response...");
        let output = model.generate(&[read_buffer.as_str().trim()], None);
        let trimmed_output = string_utils::get_uptolast(output[0].as_str(), ".", true)?;
        println!("{}", trimmed_output);
    }

    Ok(())
}

pub fn gpt2_dialog(tok_path: &str, model_path: &str) -> anyhow::Result<()> {
    let model = create_text_generation_model(tok_path, model_path)?;
    let mut done = false;
    let mut context = String::new();

    while !done {
        print!("Input: ");
        io::stdout().flush()?;
        let mut read_buffer = String::new();
        io::stdin().read_line(&mut read_buffer)?;
        done = match read_buffer.as_str().trim() {
            "q" => true,
            _ => false,
        };
        if done {
            break;
        }
        context.push_str(read_buffer.as_str().trim());

        println!("Generating response...");
        let output = model.generate(&[""], context.as_str());
        let trimmed_output = string_utils::get_uptolast(output[0].as_str(), ".", true)?;
        println!("{}", trimmed_output);

        context.push_str(trimmed_output.as_str());
    }

    Ok(())
}
