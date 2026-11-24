use color_eyre::Report;

mod handler;
mod init;

// Avoid musl's default allocator due to lackluster performance
// https://nickb.dev/blog/default-musl-allocator-considered-harmful-to-performance
#[cfg(target_env = "musl")]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

fn main() -> Result<(), color_eyre::Report> {
    let args = init::initialize()?;

    let data_dir = match args.data_dir.clone() {
        Some(dir) => dir,
        None => vecdb_dir()?,
    };
    let router = handler::get_router(data_dir)?;

    // Keep `main` synchronous; build the async runtime only for the server.
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?
        .block_on(async {
            let listener = tokio::net::TcpListener::bind(&args.listen).await?;
            tracing::info!(addr = %listener.local_addr()?, "listening");
            axum::serve(listener, router).await?;
            Ok(())
        })
}

/// Resolve where the vector DB lives.
///
/// Order: platform data dir (e.g. `%APPDATA%\rgmtrv\object-recognizer\data`,
/// `~/.local/share/object-recognizer`) → `data/` next to the executable →
/// `data/` in the current working directory.
fn vecdb_dir() -> Result<std::path::PathBuf, Report> {
    if let Some(dirs) = directories::ProjectDirs::from("my.id", "rgmtrv", "object-recognizer") {
        return Ok(dirs.data_dir().to_path_buf());
    }

    tracing::warn!("no home directory found; falling back to a local data directory");

    if let Some(exe_dir) = std::env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(|p| p.to_path_buf()))
    {
        return Ok(exe_dir.join("data"));
    }

    Ok(std::env::current_dir()?.join("data"))
}
