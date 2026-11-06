use object_recognizer_vecdb::{VECTOR_DIM, VecDb};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct Meta {
    name: String,
    category: String,
}

/// Unit-ish vector pointing mostly along axis `i`.
fn axis(i: usize) -> Vec<f32> {
    let mut v = vec![0.0; VECTOR_DIM];
    v[i] = 1.0;
    v
}

#[test]
fn add_recognize_delete_reopen() -> anyhow::Result<()> {
    let dir = std::env::temp_dir().join(format!("vecdb-test-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);

    let db = VecDb::open(&dir)?;
    let ids = db.add_with([
        (
            axis(0),
            Meta {
                name: "Red mug".into(),
                category: "kitchen".into(),
            },
        ),
        (
            axis(1),
            Meta {
                name: "Blue pen".into(),
                category: "office".into(),
            },
        ),
    ])?;
    assert_eq!(ids, vec![0, 1]);

    // exact match
    let hits = db.recognize::<Meta>(axis(0), 3, 0.8)?;
    assert_eq!(hits.len(), 1, "only the mug should pass the threshold");
    assert_eq!(hits[0].meta.name, "Red mug");
    assert!(hits[0].score > 0.99);

    // unknown object (orthogonal to everything stored) => no match
    assert!(db.recognize::<Meta>(axis(7), 3, 0.8)?.is_empty());

    // wrong dimension is rejected
    assert!(db.add(vec![vec![0.0; 3]]).is_err());

    // delete
    db.delete(&[0])?;
    assert!(db.recognize::<Meta>(axis(0), 3, 0.8)?.is_empty());
    db.optimize()?;
    drop(db);

    // reopen from disk: data + id counter survive
    let db = VecDb::open(&dir)?;
    let hits = db.recognize::<Meta>(axis(1), 3, 0.8)?;
    assert_eq!(
        hits[0].meta,
        Meta {
            name: "Blue pen".into(),
            category: "office".into()
        }
    );
    let new_ids = db.add(vec![axis(2)])?;
    assert_eq!(new_ids, vec![2], "auto-id continues after restart");

    let _ = std::fs::remove_dir_all(&dir);
    Ok(())
}
