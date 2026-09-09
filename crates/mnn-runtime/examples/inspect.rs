//! Print the named inputs and outputs of an MNN model.

use std::process::ExitCode;

use mnn_runtime::Runtime;

fn main() -> ExitCode {
    let Some(path) = std::env::args_os().nth(1) else {
        eprintln!("usage: cargo run -p mnn-runtime --example inspect -- <model.mnn>");
        return ExitCode::FAILURE;
    };

    let result = Runtime::cpu().and_then(|runtime| runtime.load_file(path));
    match result {
        Ok(model) => {
            println!("MNN {}", Runtime::native_version());
            println!("inputs:");
            for tensor in model.info().inputs() {
                println!("  {} {:?}", tensor.name(), tensor.shape());
            }
            println!("outputs:");
            for tensor in model.info().outputs() {
                println!("  {} {:?}", tensor.name(), tensor.shape());
            }
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::FAILURE
        }
    }
}
