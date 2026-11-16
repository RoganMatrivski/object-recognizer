use object_recognizer_vecdb::{all_of, any_of, eq, none_of, VecDb, VECTOR_DIM};
use serde::Deserialize;
use serde_json::json;
use std::path::PathBuf;

#[derive(Deserialize, Debug)]
struct Meta {
    name: String,
    category: String,
}

fn axis(i: usize) -> Vec<f32> {
    let mut v = vec![0.0; VECTOR_DIM];
    v[i] = 1.0;
    v
}

fn tmp(name: &str) -> PathBuf {
    let p = std::env::temp_dir().join(format!("vecdb-{}-{}", name, std::process::id()));
    let _ = std::fs::remove_dir_all(&p);
    p
}

/// mug x2 samples (kitchen, shop a), pen (office, shop a), lamp (home, shop b)
fn seeded(name: &str) -> VecDb {
    let db = VecDb::open(tmp(name)).unwrap();
    db.add_with([
        (
            axis(0),
            json!({"name":"mug","category":"kitchen","shop":"a","rank":1,"fav":true}),
        ),
        (
            axis(1),
            json!({"name":"mug","category":"kitchen","shop":"a","rank":2,"fav":false}),
        ),
        (
            axis(2),
            json!({"name":"pen","category":"office","shop":"a","rank":3,"fav":false}),
        ),
        (
            axis(3),
            json!({"name":"lamp","category":"home","shop":"b","rank":4,"fav":true}),
        ),
    ])
    .unwrap();
    db
}

#[test]
fn delete_where_removes_all_samples_of_one_object() -> anyhow::Result<()> {
    let db = seeded("w1");
    assert_eq!(db.count()?, 4);
    assert_eq!(db.delete_where("name", "mug")?, 2);
    assert_eq!(db.count()?, 2);
    assert!(db.recognize::<Meta>(axis(0), 5, 0.8)?.is_empty());
    assert!(db.recognize::<Meta>(axis(1), 5, 0.8)?.is_empty());
    assert_eq!(db.recognize::<Meta>(axis(2), 1, 0.8)?[0].meta.name, "pen");
    assert_eq!(db.delete_where("name", "does-not-exist")?, 0);
    Ok(())
}

#[test]
fn count_by_field_and_delete_by_id() -> anyhow::Result<()> {
    let db = seeded("count-by-delete-id");
    assert_eq!(
        db.count_by_field("name")?,
        vec![
            ("lamp".to_string(), 1),
            ("mug".to_string(), 2),
            ("pen".to_string(), 1),
        ]
    );
    assert!(db.delete_by_id(0)?);
    assert!(!db.delete_by_id(0)?);
    assert_eq!(db.count()?, 3);
    Ok(())
}

#[test]
fn combinators_and_field_types() -> anyhow::Result<()> {
    // AND on an indexed + an unindexed field
    let db = seeded("w2");
    assert_eq!(
        db.delete_by_filter(all_of([eq("category", "kitchen")?, eq("shop", "a")?]))?,
        2
    );

    // OR
    let db = seeded("w3");
    assert_eq!(
        db.delete_by_filter(any_of([eq("name", "pen")?, eq("name", "lamp")?]))?,
        2
    );
    assert_eq!(db.count()?, 2);

    // NOT: delete everything that is not in shop a
    let db = seeded("w4");
    assert_eq!(db.delete_by_filter(none_of([eq("shop", "a")?]))?, 1);
    assert_eq!(db.recognize::<Meta>(axis(3), 5, 0.8)?.len(), 0);

    // integer and bool equality
    let db = seeded("w5");
    assert_eq!(db.delete_where("rank", 3_i64)?, 1);
    assert_eq!(db.delete_where("fav", true)?, 2);
    assert_eq!(db.count()?, 1);
    Ok(())
}

#[test]
fn filtered_recognition_and_late_index() -> anyhow::Result<()> {
    let db = seeded("w6");
    // Query is exactly the mug vector, but restrict to category=office:
    let hits =
        db.recognize_filtered::<Meta>(axis(0), 5, -1.0, Some(all_of([eq("category", "office")?])))?;
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].meta.category, "office");

    // adding an index later on an existing db must not break filtering
    db.index_field("shop")?;
    assert_eq!(db.count_by_filter(all_of([eq("shop", "a")?]))?, 3);
    assert_eq!(db.delete_where("shop", "b")?, 1);
    Ok(())
}

#[test]
fn filters_survive_reopen() -> anyhow::Result<()> {
    let dir = tmp("w7");
    {
        let db = VecDb::open(&dir)?;
        db.add_with([(axis(0), json!({"name":"mug","category":"kitchen"}))])?;
    }
    let db = VecDb::open(&dir)?;
    assert_eq!(db.delete_where("category", "kitchen")?, 1);
    assert_eq!(db.count()?, 0);
    Ok(())
}
