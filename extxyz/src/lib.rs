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

pub fn read_frame<R>(rd: &mut R) -> Result<Frame>
where
    R: BufRead,
{
    let (natoms, info, arrs) = _read_frame(rd, None).unwrap();
    let frame = Frame { natoms, info, arrs };
    Ok(frame)
}

pub fn write_frame<W>(w: &mut W, frame: &Frame) -> Result<()>
where
    W: Write,
{
    // TODO: just put pure rust write_frame here and the test here, and leave only unsafe wrapper code in sys.
    _write_frame(w, frame.natoms, &frame.info, &frame.arrs)?;
    w.flush()?;

    Ok(())
}

#[derive(Debug)]
pub struct FrameWriter<W> {
    w: W,
}

impl<W: Write> FrameWriter<W> {
    pub fn new(w: W) -> Self {
        Self { w }
    }

    pub fn write(&mut self, frame: &Frame) -> Result<()> {
        write_frame(&mut self.w, frame)?;

        Ok(())
    }

    pub fn finish(&mut self) -> Result<()> {
        self.w.flush()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::io::{BufWriter, Cursor};

    use super::*;

    impl Frame {
        fn new_example() -> Frame {
            let inp = r#"4
key1=a key2=a/b key3=a@b key4="a@b" 
Mg        -4.25650        3.79180       -2.54123
C         -1.15405        2.86652       -1.26699
C         -5.53758        3.70936        0.63504
C         -7.28250        4.71303       -3.82016
"#;
            let mut rd = Cursor::new(inp.as_bytes());
            let frame = read_frame(&mut rd).unwrap();
            frame
        }
    }

    #[test]
    fn test_write_frame() {
        let frame = Frame::new_example();

        let mut buf = Vec::new();
        {
            let mut w = BufWriter::new(&mut buf);
            write_frame(&mut w, &frame).unwrap();
        }

        let s = String::from_utf8(buf).unwrap();
        let expect = r#"4
key1=a key2=a/b key3=a@b key4=a@b Properties=species:S:1:pos:R:3
Mg        -4.25650000          3.79180000         -2.54123000
C        -1.15405000          2.86652000         -1.26699000
C        -5.53758000          3.70936000          0.63504000
C        -7.28250000          4.71303000         -3.82016000
"#;
        assert_eq!(s, expect);
    }
}
