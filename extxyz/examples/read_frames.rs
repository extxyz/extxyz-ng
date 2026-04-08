use std::{fs};

use extxyz::read_frames;

fn main() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/examples/32768_frames.xyz");

    let file = fs::File::open(path).expect("Failed to read file");

    // wrap in Cursor for BufRead (Cursor<&[u8]> works, convert via as_bytes())
    let mut rd = std::io::BufReader::new(file);

    let frames = read_frames(&mut rd);
    dbg!(frames.count());
}
