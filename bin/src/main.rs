use color_eyre::Report;
use object_recognizer::object_recognizer_subject_cutout::Segmenter;

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

    let mut seg = Segmenter::from_path(
        object_recognizer::object_recognizer_subject_cutout::Model::U2Net,
        "../models/u2net.onnx",
    )?;
    // let mut seg = Segmenter::from_path(object_recognizer::Model::U2NetP, "../models/u2netp.onnx")?;
    println!("Loaded model in {:?}", start_total.elapsed());

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
