mod read;
mod write;

use std::io::BufRead;

use extxyz_types::Frame;

use crate::read::_read_frame_native;
pub use crate::write::write_frame;

pub type Result<T> = std::result::Result<T, ExtxyzError>;

#[derive(Debug)]
pub enum ExtxyzError {
    Io(std::io::Error),
    InvalidValue(&'static str),
}

impl std::fmt::Display for ExtxyzError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExtxyzError::Io(error) => write!(f, "{error}"),
            ExtxyzError::InvalidValue(msg) => write!(f, "{msg}"),
        }
    }
}

impl std::error::Error for ExtxyzError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            ExtxyzError::Io(error) => Some(error),
            ExtxyzError::InvalidValue(_) => None,
        }
    }
}

impl From<std::io::Error> for ExtxyzError {
    fn from(value: std::io::Error) -> Self {
        ExtxyzError::Io(value)
    }
}

pub fn read_frame<R>(rd: &mut R) -> Result<Frame>
where
    R: BufRead,
{
    // use extxyz_sys::{read_frame as _read_frame, CextxyzError};
    // let wellwelllegacy = true;
    // let frame = if wellwelllegacy {
    //     let (natoms, info, arrs) = _read_frame(rd, None)?;
    //     Frame { natoms, info, arrs }
    // } else {
    //     _read_frame_native(rd, None)?
    // };

    let frame = _read_frame_native(rd, None)?;
    Ok(frame)
}
