mod read;
mod write;

use std::io::{self, BufRead, Write};

// re-export `Frame` from extxyz_types
pub use extxyz_types::{Frame, Value};

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
    let Some(frame) = _read_frame_native(rd, None)? else {
        return Err(ExtxyzError::Io(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            "does not parse anything from reader",
        )));
    };
    Ok(frame)
}

pub struct FrameReader<'a, R> {
    // None as done marker
    rd: &'a mut R,
    finished: bool,
}

impl<'a, R> Iterator for FrameReader<'a, R>
where
    R: BufRead,
{
    type Item = Result<Frame>;

    fn next(&mut self) -> Option<Self::Item> {
        // fast finished
        if self.finished {
            return None;
        }

        match _read_frame_native(self.rd, None) {
            Ok(Some(frame)) => Some(Ok(frame)),
            Ok(None) => None,
            Err(err) => Some(Err(ExtxyzError::Io(err))),
        }
    }
}

/// read from a buf reader and return an `FrameReader` which is an interator.
pub fn read_frames<'a, R>(rd: &'a mut R) -> FrameReader<'a, R>
where
    R: BufRead,
{
    FrameReader {
        rd,
        finished: false,
    }
}

pub fn write_frames<W, I>(w: &mut W, frames: I) -> Result<()>
where
    W: Write,
    I: IntoIterator<Item = Frame>,
{
    for frame in frames {
        write_frame(w, &frame)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{BufWriter, Cursor};

    #[test]
    fn test_read_frames_default() {
        let inp = r#"4
key1=a key2=a/b key3=a@b key4="a@b"
Mg        -4.25650        3.79180       -2.54123
C         -1.15405        2.86652       -1.26699
C         -5.53758        3.70936        0.63504
C         -7.28250        4.71303       -3.82016
4
key1=a key2=a/b key3=a@b key4="a@b"
Mg        -4.25650        3.79180       -2.54123
C         -1.15405        2.86652       -1.26699
C         -5.53758        3.70936        0.63504
C         -7.28250        4.71303       -3.82016
"#;
        let mut rd = Cursor::new(inp.as_bytes());
        let mut frames = vec![];
        for frame in read_frames(&mut rd) {
            frames.push(frame);
        }

        assert_eq!(frames.len(), 2);
    }

    #[test]
    fn test_write_frames_default() {
        let inp = r#"4
key1=a key2=a/b key3=a@b key4="a@b" 
Mg        -4.25650        3.79180       -2.54123
C         -1.15405        2.86652       -1.26699
C         -5.53758        3.70936        0.63504
C         -7.28250        4.71303       -3.82016
"#;
        let rd = Cursor::new(inp.as_bytes());
        let frame1 = read_frame(&mut rd.clone()).unwrap();
        let frame2 = read_frame(&mut rd.clone()).unwrap();
        let frames = vec![frame1, frame2];
        let mut buf = Vec::new();
        {
            let mut w = BufWriter::new(&mut buf);
            write_frames(&mut w, frames).unwrap();
        }
        let s = String::from_utf8(buf).unwrap();
        let expect = r#"4
key1=a key2=a/b key3=a@b key4=a@b Properties=species:S:1:pos:R:3
Mg         -4.25650000       3.79180000      -2.54123000
C          -1.15405000       2.86652000      -1.26699000
C          -5.53758000       3.70936000       0.63504000
C          -7.28250000       4.71303000      -3.82016000
4
key1=a key2=a/b key3=a@b key4=a@b Properties=species:S:1:pos:R:3
Mg         -4.25650000       3.79180000      -2.54123000
C          -1.15405000       2.86652000      -1.26699000
C          -5.53758000       3.70936000       0.63504000
C          -7.28250000       4.71303000      -3.82016000
"#;
        assert_eq!(s, expect);
    }
}
