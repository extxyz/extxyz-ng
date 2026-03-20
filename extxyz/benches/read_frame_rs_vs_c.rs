use std::io::{BufRead, BufReader, Cursor};

use criterion::{criterion_group, criterion_main, Criterion};
use extxyz::{read_frame, Frame};

pub fn read_frame_c_binding<R>(rd: &mut R) -> Frame
where
    R: BufRead,
{
    use extxyz_sys::read_frame as _read_frame;
    let (natoms, info, arrs) = _read_frame(rd, None).unwrap();

    Frame { natoms, info, arrs }
}

fn bench_read_frame(c: &mut Criterion) {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/benches/6M0J.xyz");
    let data = std::fs::read(path).unwrap();
    let mut group = c.benchmark_group("read_frame");

    group.bench_function("read_frame_rs", |b| {
        b.iter(|| {
            let cursor = Cursor::new(&data);
            let mut rd = BufReader::new(cursor);
            let _ = read_frame(&mut rd).unwrap();
        })
    });

    group.bench_function("read_frame_c_binding", |b| {
        b.iter(|| {
            let cursor = Cursor::new(&data);
            let mut rd = BufReader::new(cursor);
            let _ = read_frame_c_binding(&mut rd);
        })
    });
    group.finish();
}

criterion_group!(benches, bench_read_frame);
criterion_main!(benches);
