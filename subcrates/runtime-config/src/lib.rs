use std::env;

fn cores() -> usize {
    // physical cores, falling back to logical
    let phys = num_cpus::get_physical();
    let logical = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(phys);
    phys.min(logical).max(1)
}

fn env_threads(key: &str) -> Option<usize> {
    env::var(key).ok()?.parse().ok().filter(|&n: &usize| n >= 1)
}

/// Heavy model: give it most cores, keep 1 for search, UI and the rest.
pub fn segmenter_threads() -> usize {
    env_threads("ORT_SEG_THREADS").unwrap_or_else(|| cores().saturating_sub(1).clamp(1, 4))
}

/// Runs right before the search: stay small so the search workers get a core.
pub fn embedder_threads() -> usize {
    env_threads("ORT_EMB_THREADS").unwrap_or_else(|| (cores() / 2).clamp(1, 2))
}
