use flate2::{Compression, read::GzDecoder, write::GzEncoder};
use object_recognizer_vecdb::{ImportMode, VECTOR_DIM, VecDb};
use serde::{Deserialize, Serialize};
use std::{
    io::{Read, Write},
    path::PathBuf,
};

#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct Meta {
    name: String,
}

fn axis(i: usize) -> Vec<f32> {
    let mut v = vec![0.0; VECTOR_DIM];
    v[i] = 1.0;
    v
}

fn tmp(name: &str) -> PathBuf {
    let p = std::env::temp_dir().join(format!("vecdb-{}-{}", name, std::process::id()));
    let _ = std::fs::remove_dir_all(&p);
    let _ = std::fs::remove_file(&p);
    p
}

fn meta(n: &str) -> Meta {
    Meta { name: n.into() }
}

#[test]
fn export_import_between_machines() -> anyhow::Result<()> {
    // "machine A"
    let a = VecDb::open(tmp("a"))?;
    a.add_with([
        (axis(0), meta("mug")),
        (axis(1), meta("pen")),
        (axis(2), meta("lamp")),
    ])?;
    a.delete(&[1])?; // exported data must not include deleted points
    let file = tmp("share.vecdb");
    assert_eq!(a.export(&file)?, 2);

    // "machine B": fresh db -> merge
    let b = VecDb::open(tmp("b"))?;
    assert_eq!(b.import(&file, ImportMode::Merge)?, 2);
    assert_eq!(b.recognize::<Meta>(axis(0), 1, 0.8)?[0].meta, meta("mug"));
    assert_eq!(b.recognize::<Meta>(axis(2), 1, 0.8)?[0].meta, meta("lamp"));
    assert!(
        b.recognize::<Meta>(axis(1), 1, 0.8)?.is_empty(),
        "deleted 'pen' must not travel"
    );

    // merge into a db that already has data: ids must not collide
    let c = VecDb::open(tmp("c"))?;
    c.add_with([(axis(5), meta("local-thing"))])?; // takes id 0, same as 'mug' in the file
    c.import(&file, ImportMode::Merge)?;
    assert_eq!(
        c.recognize::<Meta>(axis(5), 1, 0.8)?[0].meta,
        meta("local-thing")
    );
    assert_eq!(c.recognize::<Meta>(axis(0), 1, 0.8)?[0].meta, meta("mug"));

    // replace: local knowledge is dropped, file ids preserved, counter continues after them
    let d = VecDb::open(tmp("d"))?;
    d.add_with([(axis(9), meta("old"))])?;
    d.import(&file, ImportMode::Replace)?;
    assert!(d.recognize::<Meta>(axis(9), 1, 0.8)?.is_empty());
    assert_eq!(d.recognize::<Meta>(axis(0), 1, 0.8)?[0].id, 0);
    assert_eq!(d.recognize::<Meta>(axis(2), 1, 0.8)?[0].id, 2);
    assert_eq!(d.add(vec![axis(3)])?, vec![3]);
    Ok(())
}

#[test]
fn incompatible_file_is_rejected_and_db_untouched() -> anyhow::Result<()> {
    let a = VecDb::open(tmp("e"))?;
    a.add_with([(axis(0), meta("mug"))])?;
    let file = tmp("ok.vecdb");
    a.export(&file)?;

    // tamper with the model name in the header
    let mut text = String::new();
    GzDecoder::new(std::fs::File::open(&file)?).read_to_string(&mut text)?;
    let bad = text.replacen("my-embedder-v1", "other-model", 1);
    let bad_file = tmp("bad.vecdb");
    let mut enc = GzEncoder::new(std::fs::File::create(&bad_file)?, Compression::default());
    enc.write_all(bad.as_bytes())?;
    enc.finish()?;

    let b = VecDb::open(tmp("f"))?;
    b.add_with([(axis(4), meta("keep-me"))])?;
    let err = b
        .import(&bad_file, ImportMode::Replace)
        .unwrap_err()
        .to_string();
    assert!(err.contains("model mismatch"), "got: {err}");
    assert_eq!(
        b.recognize::<Meta>(axis(4), 1, 0.8)?[0].meta,
        meta("keep-me")
    );
    Ok(())
}
