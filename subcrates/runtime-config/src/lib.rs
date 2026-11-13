pub fn intra_threads() -> usize {
    std::env::var("ORT_INTRA_THREADS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or_else(|| num_cpus::get().min(4))
}
