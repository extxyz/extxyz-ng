use std::fs;

use extxyz::read_frame;

fn main() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/benches/6VXX.xyz");

    let file = fs::File::open(path).expect("Failed to read file");
    let mut rd = std::io::BufReader::new(file);

    let frame = read_frame(&mut rd).unwrap();
    dbg!(frame.natoms());
}
