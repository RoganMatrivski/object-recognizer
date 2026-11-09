use color_eyre::Report;
use object_recognizer::object_recognizer_feature_extraction::FeatureExtractor;
use object_recognizer::object_recognizer_subject_cutout::{Model, Segmenter};

mod init;

// Avoid musl's default allocator due to lackluster performance
// https://nickb.dev/blog/default-musl-allocator-considered-harmful-to-performance
#[cfg(target_env = "musl")]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

#[tracing::instrument]
fn main() -> Result<(), Report> {
    init::initialize()?;
    println!("Hello, world!");

    let start_total = std::time::Instant::now();

    let samples_dir = "D:/fruit-recognizer-py/samples";

    let _env = ort::init().build()?;

    let mut seg = Segmenter::new_with_env(Model::U2Net, &_env, |env| {
        ort::session::Session::builder(env)?.commit_from_memory(include_bytes!(
            "../../subcrates/subject-cutout/models/u2net.onnx"
        ))
    })?;

    let mut extractor = FeatureExtractor::new_embedded_with_env(&_env)?;

    println!("Loaded models in {:?}", start_total.elapsed());

    for entry in std::fs::read_dir(samples_dir)? {
        let entry = entry?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }

        let file_name = match path.file_name().and_then(|n| n.to_str()) {
            Some(name) => name.to_string(),
            None => continue,
        };

        let file_start = std::time::Instant::now();
        println!("Processing file: {}", file_name);

        let img = match image::open(&path) {
            Ok(img) => img,
            Err(e) => {
                eprintln!("Failed to open image {}: {:?}", file_name, e);
                continue;
            }
        };

        let seg_start = std::time::Instant::now();
        let outs = seg.instance_cutout(&img)?;
        println!("  - Segmentation took {:?}", seg_start.elapsed());

        let extract_start = std::time::Instant::now();
        let mut embeddings = Vec::new();
        for out in &outs {
            let dyn_img = image::DynamicImage::ImageRgba8(out.clone());
            let embedding = extractor.extract(&dyn_img)?;
            embeddings.push(embedding);
        }
        println!(
            "  - Feature extraction for {} cutout(s) took {:?}",
            outs.len(),
            extract_start.elapsed()
        );

        let stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("output");
        std::fs::create_dir_all("tmp")?;

        let save_start = std::time::Instant::now();
        for (i, out) in outs.into_iter().enumerate() {
            out.save(format!("tmp/{stem} {i}.png"))?;
        }
        println!("  - Saving cutouts took {:?}", save_start.elapsed());
        println!(
            "Finished file {} in total {:?}",
            file_name,
            file_start.elapsed()
        );
    }

    println!("All processing completed in {:?}", start_total.elapsed());

    Ok(())
}
