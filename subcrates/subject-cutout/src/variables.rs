// ---------------------------------------------------------------------------
// Tuning knobs. Defaults were picked by eye; adjust if your images disagree.
// ---------------------------------------------------------------------------

/// Upper cap on the Otsu threshold (0-255) used to binarize the saliency map.
/// Lower = keeps more weakly-salient areas (fewer holes / missing highlights, but
/// more background sneaks in). Higher = stricter and cleaner, but may lose faint parts.
/// Try 80-140.
pub const OTSU_CAP: u8 = 110;

/// Crack-bridging radius is `shorter_side / CLOSE_DIVISOR` (model pixels, min 1).
/// Smaller divisor = bigger radius: bridges wider gaps, but may merge subjects that
/// sit close together. Larger divisor = subjects split more easily. Try 50-200.
pub const CLOSE_DIVISOR: u32 = 100;

/// Subjects smaller than `1 / MIN_AREA_DIVISOR` of the image area are dropped as specks.
/// 200 = 0.5%. Lower = keeps small objects (and some noise); higher = main subjects
/// only. Try 50-2000.
pub const MIN_AREA_DIVISOR: u32 = 200;

/// If the model's output spans less than this (output is expected in 0..1), the image
/// is treated as having no subject. Stops min-max normalization from inflating noise
/// into fake subjects. If real subjects vanish, lower it (or check your model's
/// output scale).
pub const MIN_SALIENCY_RANGE: f32 = 0.1;

/// Radius (model pixels) of the soft-edge band: how far past the binary mask the
/// model's own soft edge may extend, and how far inside the alpha is forced opaque.
/// The per-subject crop is padded by the same amount. Keep it small (1-2).
pub const EDGE_RADIUS: u8 = 1;

/// Edge ramp is remapped with a smoothstep between these fractions of full alpha.
/// Narrower (e.g. 0.35 / 0.65) = crisper edge; wider (e.g. 0.1 / 0.9) = softer.
pub const EDGE_LO: f32 = 0.2;
pub const EDGE_HI: f32 = 0.8;

/// Colors of semi-transparent edge pixels are replaced with colors from the subject's
/// opaque interior, to avoid halos from the old background. A pixel keeps its original
/// color in proportion to `(alpha/255)^BLEED_KEEP_POW`. Higher = keeps more original
/// color on nearly-opaque pixels (subtler); 1 = linear; very high = replace almost all.
pub const BLEED_KEEP_POW: i32 = 4;

/// Longest side (in cells) the color-fill grid may have. Only a speed/memory cap.
pub const BLEED_GRID_MAX: u32 = 512;

/// Only used with the `jpeg-scaled` feature. JPEGs are decoded at the largest
/// 1/2, 1/4 or 1/8 reduction whose shorter side is still >= this (pixels). Cutouts
/// come out at this reduced size, so raise it if you need bigger crops. The model
/// itself only sees 256x256, so ~640 leaves headroom. 0 is not allowed: use >= 1.
pub const JPEG_MIN_SIDE: u16 = 640;

/// Background colour the cutouts are flattened onto when squared up (`square_on`,
/// `instance_squares`). Default is the ImageNet mean (0.485, 0.456, 0.406) * 255, which
/// normalizes to exactly 0 for ImageNet-style extractors, so the padding carries no
/// signal. Change it if your extractor uses different mean/std (e.g. CLIP).
pub const SQUARE_BG: [u8; 3] = [124, 116, 104];
