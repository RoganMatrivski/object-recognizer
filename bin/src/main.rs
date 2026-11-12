use color_eyre::Report;

mod init;

// Avoid musl's default allocator due to lackluster performance
// https://nickb.dev/blog/default-musl-allocator-considered-harmful-to-performance
#[cfg(target_env = "musl")]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

static TEST_DATA: [(&[u8], &str); 1] = [
    (&[0u8], "asdf"), // (include_bytes!("../testdata/apple (6).jpg"), "apple (6).jpg"),
                      // (
                      //     include_bytes!("../testdata/bananas (3).jpg"),
                      //     "bananas (3).jpg",
                      // ),
                      // (
                      //     include_bytes!("../testdata/durian (16).jpg"),
                      //     "durian (16).jpg",
                      // ),
                      // (
                      //     include_bytes!("../testdata/durian (5).jpg"),
                      //     "durian (5).jpg",
                      // ),
                      // (
                      //     include_bytes!("../testdata/durians (3).jpg"),
                      //     "durians (3).jpg",
                      // ),
                      // (
                      //     include_bytes!("../testdata/grape - 1 (6).jpg"),
                      //     "grape - 1 (6).jpg",
                      // ),
                      // (
                      //     include_bytes!("../testdata/grape - 2 (4).jpg"),
                      //     "grape - 2 (4).jpg",
                      // ),
                      // (
                      //     include_bytes!("../testdata/grapes (3).jpg"),
                      //     "grapes (3).jpg",
                      // ),
                      // (
                      //     include_bytes!("../testdata/green apple (6).jpg"),
                      //     "green apple (6).jpg",
                      // ),
                      // (
                      //     include_bytes!("../testdata/green peppers (6).jpg"),
                      //     "green peppers (6).jpg",
                      // ),
                      // (
                      //     include_bytes!("../testdata/orange (4).jpg"),
                      //     "orange (4).jpg",
                      // ),
                      // (
                      //     include_bytes!("../testdata/oranges (4).jpg"),
                      //     "oranges (4).jpg",
                      // ),
                      // (
                      //     include_bytes!("../testdata/papaya (7).jpg"),
                      //     "papaya (7).jpg",
                      // ),
                      // (include_bytes!("../testdata/pear (4).jpg"), "pear (4).jpg"),
                      // (include_bytes!("../testdata/pears (6).jpg"), "pears (6).jpg"),
                      // (
                      //     include_bytes!("../testdata/red peppers (5).jpg"),
                      //     "red peppers (5).jpg",
                      // ),
];

#[tracing::instrument]
fn main() -> Result<(), color_eyre::Report> {
    let _args = init::initialize()?;
    println!("Hello, world!");

    let start_total = std::time::Instant::now();
    let mut recognizer = object_recognizer::Recognizer::new_embedded(tempfile::tempdir()?.path())?;

    for (data, name) in TEST_DATA {
        let start_img = std::time::Instant::now();
        let img = image::load_from_memory(&data)?;
        let cutouts = recognizer.segment_subject(&img)?;
        for cutout in &cutouts {
            let _embedding = recognizer.generate_embedding(cutout)?;
        }
        tracing::trace!(name = %name, cutouts_count = cutouts.len(), elapsed_ms = ?start_img.elapsed().as_secs_f64() * 1000.0, "processed test image with embedding");
    }

    tracing::trace!(elapsed_ms = ?start_total.elapsed().as_secs_f64() * 1000.0, "full test suite completed");

    Ok(())
}
