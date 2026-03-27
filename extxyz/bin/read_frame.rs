use std::{fs, io::Cursor};

use extxyz::read_frame;

fn main() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/benches/6VXX.xyz");

    // read entire file into a String
    let content = fs::read_to_string(path).expect("Failed to read file");

    // get a &str reference
    let s = &content;

    // wrap in Cursor for BufRead (Cursor<&[u8]> works, convert via as_bytes())
    let mut rd = std::io::BufReader::new(Cursor::new(s.as_bytes()));

    let frame = read_frame(&mut rd).unwrap();
    dbg!(frame.natoms());
}
