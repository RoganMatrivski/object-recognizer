use crate::RecognizerWrapper;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

fn dummy_image_bytes() -> Vec<u8> {
    let img = image::RgbImage::new(100, 100);
    let mut bytes = Vec::new();
    let mut cursor = std::io::Cursor::new(&mut bytes);
    img.write_to(&mut cursor, image::ImageFormat::Jpeg).unwrap();
    bytes
}

#[test]
fn test_search_not_blocked_by_classify() {
    let wrapper = Arc::new(RecognizerWrapper::new_embedded_default().unwrap());
    let img = dummy_image_bytes();

    wrapper
        .vector_db_insert(
            vec![0.0; 1280],
            serde_json::json!({"name": "test"}).to_string(),
        )
        .unwrap();

    let running = Arc::new(AtomicBool::new(true));
    let r_clone = Arc::clone(&running);
    let w_classify = Arc::clone(&wrapper);
    let img_clone = img.clone();

    let classify_handle = std::thread::spawn(move || {
        while r_clone.load(Ordering::Relaxed) {
            let _ = w_classify.classify_image(img_clone.clone(), 1, 0.0);
        }
    });

    let w_search = Arc::clone(&wrapper);
    let search_handle = std::thread::spawn(move || {
        let mut latencies = vec![];
        for _ in 0..50 {
            let start = Instant::now();
            let _ = w_search.vector_db_search(vec![0.0; 1280], 1).unwrap();
            latencies.push(start.elapsed());
            std::thread::sleep(Duration::from_millis(5));
        }
        latencies
    });

    std::thread::sleep(Duration::from_millis(500));
    running.store(false, Ordering::Relaxed);
    classify_handle.join().unwrap();
    let latencies = search_handle.join().unwrap();

    let mut sorted = latencies;
    sorted.sort();
    let p99 = sorted[(sorted.len() * 99 / 100).min(sorted.len() - 1)];
    assert!(
        p99 < Duration::from_millis(50),
        "p99 search latency was too high: {p99:?}"
    );
}

#[test]
fn test_soak_concurrency() {
    let wrapper = Arc::new(RecognizerWrapper::new_embedded_default().unwrap());
    let img = dummy_image_bytes();
    let running = Arc::new(AtomicBool::new(true));
    let mut handles = vec![];

    let r1 = Arc::clone(&running);
    let w1 = Arc::clone(&wrapper);
    let img1 = img.clone();
    handles.push(std::thread::spawn(move || {
        while r1.load(Ordering::Relaxed) {
            let _ = w1.generate_embedding(img1.clone());
        }
    }));

    let r2 = Arc::clone(&running);
    let w2 = Arc::clone(&wrapper);
    let img2 = img.clone();
    handles.push(std::thread::spawn(move || {
        while r2.load(Ordering::Relaxed) {
            let _ = w2.segment_subject_raw(img2.clone());
        }
    }));

    let r3 = Arc::clone(&running);
    let w3 = Arc::clone(&wrapper);
    let img3 = img.clone();
    handles.push(std::thread::spawn(move || {
        while r3.load(Ordering::Relaxed) {
            let _ = w3.add_training_sample(img3.clone(), "{\"cat\": \"test\"}".into());
        }
    }));

    let r4 = Arc::clone(&running);
    let w4 = Arc::clone(&wrapper);
    handles.push(std::thread::spawn(move || {
        while r4.load(Ordering::Relaxed) {
            let _ = w4.vector_db_search(vec![0.0; 1280], 1);
        }
    }));

    std::thread::sleep(Duration::from_secs(2));
    running.store(false, Ordering::Relaxed);
    for h in handles {
        h.join().unwrap();
    }
}
