pub const MODEL_WIDTH: u32 = 256;
pub const MODEL_HEIGHT: u32 = 256;
pub const EMBEDDING_DIM: usize = 1280;
pub const EMBEDDED_MODEL_BYTES: &[u8] = include_bytes!("../model/mobilenetv4_embedding.onnx");
