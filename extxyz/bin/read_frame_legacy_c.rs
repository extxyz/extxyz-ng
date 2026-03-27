use std::{
    fs,
    io::{BufRead, Cursor},
};

use extxyz::Frame;

pub fn read_frame_c_binding<R>(rd: &mut R) -> Frame
where
    R: BufRead,
{
    #[cfg(feature = "legacy")]
    use extxyz_sys::read_frame as _read_frame;
    let (natoms, info, arrs) = _read_frame(rd, None).unwrap();

    Frame { natoms, info, arrs }
}

fn main() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/benches/6VXX.xyz");

    // read entire file into a String
    let content = fs::read_to_string(path).expect("Failed to read file");

    // get a &str reference
    let s = &content;

    // wrap in Cursor for BufRead (Cursor<&[u8]> works, convert via as_bytes())
    let mut rd = std::io::BufReader::new(Cursor::new(s.as_bytes()));

    let frame = read_frame_c_binding(&mut rd);
    dbg!(frame.natoms());
}
