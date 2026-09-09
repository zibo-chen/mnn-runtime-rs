//! Run one inference with zero-filled inputs.

use std::process::ExitCode;

use mnn_runtime::{Backend, Runtime, RuntimeConfig, Tensor};

fn main() -> ExitCode {
    let Some(path) = std::env::args_os().nth(1) else {
        eprintln!("usage: cargo run -p mnn-runtime --example smoke -- <model.mnn>");
        return ExitCode::FAILURE;
    };

    let result = (|| {
        let backend = match std::env::var("MNN_SMOKE_BACKEND").as_deref() {
            Ok("metal") => Backend::Metal,
            Ok("coreml") => Backend::CoreMl,
            Ok("opencl") => Backend::OpenCl,
            Ok("opengl") => Backend::OpenGl,
            Ok("vulkan") => Backend::Vulkan,
            Ok("auto") => Backend::Auto,
            Ok(_) | Err(_) => Backend::Cpu,
        };
        let runtime = Runtime::new(RuntimeConfig::new().with_backend(backend))?;
        let model = runtime.load_file(path)?;
        let inputs = model
            .info()
            .inputs()
            .iter()
            .map(|info| {
                Tensor::new(
                    info.name(),
                    info.shape().to_vec(),
                    vec![0.0; info.element_count()],
                )
            })
            .collect::<mnn_runtime::Result<Vec<_>>>()?;
        model.run(&inputs)
    })();

    match result {
        Ok(outputs) => {
            println!("MNN {}", Runtime::native_version());
            for tensor in outputs {
                let first = tensor.data().first().copied().unwrap_or_default();
                println!(
                    "{} {:?}: {} values, first={first}",
                    tensor.name(),
                    tensor.shape(),
                    tensor.data().len()
                );
            }
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::FAILURE
        }
    }
}
