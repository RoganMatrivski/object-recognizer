use color_eyre::Report;
use object_recognizer_vecdb::VecDb;

use eyre::Context;

mod init;

// Avoid musl's default allocator due to lackluster performance
// https://nickb.dev/blog/default-musl-allocator-considered-harmful-to-performance
#[cfg(target_env = "musl")]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

static DIM: usize = 1280;
static COUNT: usize = 32;

#[tracing::instrument]
fn main() -> Result<(), Report> {
    init::initialize()?;

    let dirpath = tempfile::tempdir()?;
    tracing::info!("Initializing db at {:?}", dirpath.path());

    let db = VecDb::open(dirpath.path()).unwrap();

    tracing::info!("Filling db with random {DIM}-dim vectors");
    let vectors = (0..COUNT)
        .map(|_| rand::random::<[f32; DIM]>().to_vec())
        .collect::<Vec<_>>();

    db.add(vectors).expect("Failed to setup db");

    tracing::info!("Setup done!");

    let query = rand::random::<[f32; DIM]>().to_vec();

    let start = std::time::Instant::now();
    let iterations = 128;
    for _ in 0..iterations {
        let _ = db.search(query.clone(), 5).expect("Failed to do query");
    }
    let duration = start.elapsed();

    tracing::info!(
        "Completed {} searches in {:?} ({:?} per search)",
        iterations,
        duration,
        duration / iterations
    );

    Ok(())
}
