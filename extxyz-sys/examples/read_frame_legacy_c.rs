use extxyz_types::Frame;
use std::{fs, io::BufRead};

pub fn read_frame_c_binding<R>(rd: &mut R) -> Frame
where
    R: BufRead,
{
    use extxyz_sys::read_frame as _read_frame;
    let (natoms, info, arrs) = _read_frame(rd, None).unwrap();

    Frame::new(natoms, info.0, arrs.0)
}

fn main() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/benches/6VXX.xyz");

    let file = fs::File::open(path).expect("Failed to read file");

    // wrap in Cursor for BufRead (Cursor<&[u8]> works, convert via as_bytes())
    let mut rd = std::io::BufReader::new(file);

    let frame = read_frame_c_binding(&mut rd);
    dbg!(frame.natoms());
}
