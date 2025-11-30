use std::io::{BufRead, Write};

use extxyz_sys::{
    read_frame as _read_frame, write_frame as _write_frame, CextxyzError, DictHandler,
};

pub type Result<T> = std::result::Result<T, ExtxyzError>;

#[derive(Debug)]
pub enum ExtxyzError {
    Io(std::io::Error),
    WrapperError(CextxyzError),
}

impl From<std::io::Error> for ExtxyzError {
    fn from(value: std::io::Error) -> Self {
        ExtxyzError::Io(value)
    }
}

impl From<CextxyzError> for ExtxyzError {
    fn from(value: CextxyzError) -> Self {
        ExtxyzError::WrapperError(value)
    }
}

pub struct Frame {
    natoms: u32,
    info: DictHandler,
    arrs: DictHandler,
}

pub fn read_frame<R: BufRead>(rd: R) -> Result<Frame> {
    let (natoms, info, arrs) = _read_frame(rd, None).unwrap();
    let frame = Frame { natoms, info, arrs };
    Ok(frame)
}

pub fn write_frame<W: Write>(w: &mut W, frame: &Frame) -> Result<()> {
    _write_frame(w, frame.natoms, &frame.info, &frame.arrs)?;
    Ok(())
}

// TODO: just put pure rust write_frame here and the test here, and leave only unsafe wrapper code in sys.
