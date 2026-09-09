//! Capture all OCR model outputs with identical inputs, for separate-process comparison.
use ocr_rs::{Backend, InferenceConfig, InferenceEngine};
use std::{fs, io::Write, path::PathBuf};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    assert_eq!(
        args.len(),
        2,
        "usage: probe <models-directory> <output-directory>"
    );
    let output_dir = PathBuf::from(&args[1]);
    fs::create_dir_all(&output_dir)?;
    let mut models: Vec<_> = fs::read_dir(&args[0])?
        .map(|entry| entry.unwrap().path())
        .filter(|p| p.extension().is_some_and(|e| e == "mnn"))
        .collect();
    models.sort();
    assert_eq!(models.len(), 24, "expected the full OCR model corpus");
    let backend = match std::env::var("PROBE_BACKEND").as_deref() {
        Ok("metal") => Backend::Metal,
        Ok("opencl") => Backend::OpenCL,
        Ok("vulkan") => Backend::Vulkan,
        Ok("cuda") => Backend::CUDA,
        _ => Backend::CPU,
    };
    let config = InferenceConfig::new().with_threads(4).with_backend(backend);
    let mut cases = 0;
    for path in models {
        let name = path.file_stem().unwrap().to_str().unwrap();
        let engine = InferenceEngine::from_file(&path, Some(config.clone()))?;
        let shapes: Vec<Vec<usize>> = if name.contains("_det") {
            vec![vec![1, 3, 64, 96], vec![1, 3, 96, 160], vec![1, 3, 64, 96]]
        } else if name.contains("_rec") {
            vec![
                vec![1, 3, 48, 64],
                vec![3, 3, 48, 160],
                vec![1, 3, 48, 96],
                vec![1, 3, 48, 64],
            ]
        } else {
            vec![vec![1, 3, 224, 224]]
        };
        for (case, shape) in shapes.into_iter().enumerate() {
            let values: Vec<_> = (0..shape.iter().product())
                .map(|i: usize| ((i % 251) as f32 - 125.0) / 128.0)
                .collect();
            let (output, actual_shape) = engine.run_dynamic_raw(&values, &shape)?;
            assert!(!output.is_empty());
            assert!(
                output.iter().all(|x| x.is_finite()),
                "{name} returned a nonfinite value"
            );
            let mut file = fs::File::create(output_dir.join(format!("{name}.{case}.bin")))?;
            file.write_all(&(actual_shape.len() as u32).to_le_bytes())?;
            for dim in actual_shape {
                file.write_all(&(dim as u64).to_le_bytes())?;
            }
            for value in output {
                file.write_all(&value.to_le_bytes())?;
            }
            cases += 1;
        }
        println!("validated {name}");
    }
    println!("captured {cases} inference cases from 24 models");
    Ok(())
}
