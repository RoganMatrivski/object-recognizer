//! Fail the build if an embedded model is a Git LFS pointer instead of the
//! real file. Otherwise `include_bytes!` embeds the ~130-byte pointer and the
//! app only fails at startup with "protobuf parsing failed".

fn main() {
    for (feature, path) in [
        ("CARGO_FEATURE_EMBED_U2NET", "models/u2net_int8_256.onnx"),
        ("CARGO_FEATURE_EMBED_U2NETP", "models/u2netp.onnx"),
    ] {
        println!("cargo:rerun-if-changed={path}");
        if std::env::var_os(feature).is_some() {
            check_not_lfs_pointer(path);
        }
    }
}

fn check_not_lfs_pointer(path: &str) {
    let head = std::fs::read(path).unwrap_or_else(|e| panic!("cannot read {path}: {e}"));
    if head.starts_with(b"version https://git-lfs") {
        panic!("{path} is a Git LFS pointer, not the model. Run `git lfs pull` and rebuild.");
    }
}
