mod read;

use std::io::{BufRead, Write};

use extxyz_sys::{read_frame as _read_frame, CextxyzError};
use extxyz_types::{escape, Frame, Value};

pub type Result<T> = std::result::Result<T, ExtxyzError>;

#[derive(Debug)]
pub enum ExtxyzError {
    Io(std::io::Error),
    WrapperError(CextxyzError),
    InvalidValue(&'static str),
}

impl std::fmt::Display for ExtxyzError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExtxyzError::Io(error) => write!(f, "{error}"),
            ExtxyzError::WrapperError(cextxyz_error) => write!(f, "{cextxyz_error}"),
            ExtxyzError::InvalidValue(msg) => write!(f, "{msg}"),
        }
    }
}

impl std::error::Error for ExtxyzError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            ExtxyzError::Io(error) => Some(error),
            ExtxyzError::WrapperError(cextxyz_error) => Some(cextxyz_error),
            ExtxyzError::InvalidValue(_) => None,
        }
    }
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

pub fn read_frame<R>(rd: &mut R) -> Result<Frame>
where
    R: BufRead,
{
    let (natoms, info, arrs) = _read_frame(rd, None)?;
    let frame = Frame { natoms, info, arrs };
    Ok(frame)
}

fn write_lattice<T, W>(w: &mut W, m: &[Vec<T>]) -> Result<()>
where
    T: Default + std::fmt::Display + Copy,
    W: Write,
{
    if m.len() != 3 {
        return Err(ExtxyzError::InvalidValue("expect 3x3 matrix"));
    }

    // transpose lattice matrix which has vectors in column-wise.
    let mut m3 = [[T::default(); 3]; 3];

    for i in 0..3 {
        if m[i].len() != 3 {
            return Err(ExtxyzError::InvalidValue("expect 3x3 matrix"));
        }
        for j in 0..3 {
            m3[i][j] = m[j][i];
        }
    }

    write!(w, "\"")?;

    m3.as_flattened()
        .iter()
        .try_for_each(|s| write!(w, "{s}"))?;

    write!(w, "\"")?;

    Ok(())
}

fn write_vec<T, W>(w: &mut W, s: &[T]) -> Result<()>
where
    T: std::fmt::Display,
    W: Write,
{
    let indent = " ".repeat(4);
    let s = s
        .iter()
        .map(|i| format!("{i}"))
        .collect::<Vec<_>>()
        .join(&indent);
    write!(w, "{s}")?;
    Ok(())
}

/// instead of calling c api, it is easier to reimplement it, because I don't need to do parsing.
/// since it is rust, performance wise also compatible with c implementation and safe.
///
/// # Errors
/// ???
pub fn write_frame<W>(w: &mut W, frame: &Frame) -> Result<()>
where
    W: Write,
{
    let natoms = frame.natoms();

    writeln!(w, "{natoms}")?;

    // info
    let mut iter = frame.info.0.iter().peekable();
    while let Some((k, v)) = iter.next() {
        // the inner datastructure will store "Properties" as a key (if exist), but in the
        // write function the Properties field is deduct from the arr.
        // When read the xyz may not have "Properties" field, but write will always have it.
        // XXX: therefore in the read (the parser, if I impl myself) need to validate the
        // properties is conform with what provided.
        if k.as_str() == "Properties" {
            continue;
        }

        let s = escape(k);
        write!(w, "{s}")?;
        write!(w, "=")?;

        // in extxyz c implementation, lattice treated different write in column-wise and use
        // single space as spliter
        if k.as_str() == "Lattice" {
            // XXX: check in the parser, whether is the Lattice can have item type as Integer of Bool?
            // From perspective of a crystal parser, the spec should be more strict that it is
            // always parsed as Float.
            match v {
                Value::MatrixInteger(m, _) => {
                    write_lattice(w, m)?;
                }
                Value::MatrixFloat(m, _) => {
                    write_lattice(w, m)?;
                }
                Value::MatrixBool(m, _) => {
                    write_lattice(w, m)?;
                }
                _ => {
                    // this is unreachable if the inner dict is not create manually
                    return Err(ExtxyzError::InvalidValue(
                        "Lattice must be a 3x3 int/float matrix",
                    ));
                }
            }
        } else {
            write!(w, "{v}")?;
        }

        // only add a space if there is more to print in info array
        if iter.peek().is_some() {
            write!(w, " ")?;
        }
    }

    // "Properties" deduct from the arrs
    write!(w, " ")?;
    write!(w, "Properties=")?;

    let mut s = String::new();
    let mut iter = frame.arrs.0.iter().peekable();
    // loop over columns, so it cant be single value but must be vec, or matrix
    while let Some((k, v)) = iter.next() {
        s.push_str(k);
        s.push(':');
        match v {
            Value::VecInteger(_, _) => s.push_str("I:1"),
            Value::VecFloat(_, _) => s.push_str("R:1"),
            Value::VecBool(_, _) => s.push_str("L:1"),
            Value::VecText(_, _) => s.push_str("S:1"),
            Value::MatrixInteger(_, shape) => s.push_str(format!("I:{}", shape.1).as_str()),
            Value::MatrixFloat(_, shape) => s.push_str(format!("R:{}", shape.1).as_str()),
            Value::MatrixBool(_, shape) => s.push_str(format!("L:{}", shape.1).as_str()),
            Value::MatrixText(_, shape) => s.push_str(format!("S:{}", shape.1).as_str()),
            _ => {
                // this is unreachable if the inner dict is not create manually
                return Err(ExtxyzError::InvalidValue(
                    "arrs can only be vector or matrix",
                ));
            }
        }

        if iter.peek().is_some() {
            s.push(':');
        }
    }
    write!(w, "{}", escape(&s))?;
    writeln!(w)?;

    // out loop, over atoms (rows)
    for i in 0..natoms {
        let mut iter = frame.arrs.0.iter().peekable();
        // loop over by columns (property in properties list)
        while let Some((_, v)) = iter.next() {
            let i = i as usize;

            // store the columns but write row by row
            match v {
                Value::VecInteger(items, _) => write!(w, "{}", items[i])?,
                Value::VecFloat(items, _) => write!(w, "{}", items[i])?,
                Value::VecBool(items, _) => write!(w, "{}", items[i])?,
                Value::VecText(items, _) => write!(w, "{}", items[i])?,
                Value::MatrixInteger(items, _) => {
                    let s = &items[i];
                    write_vec(w, s)?;
                }
                Value::MatrixFloat(items, _) => {
                    let s = &items[i];
                    write_vec(w, s)?;
                }
                Value::MatrixBool(items, _) => {
                    let s = &items[i];
                    write_vec(w, s)?;
                }
                Value::MatrixText(items, _) => {
                    let s = &items[i];
                    write_vec(w, s)?;
                }
                _ => {
                    // this is unreachable if the inner dict is not create manually
                    return Err(ExtxyzError::InvalidValue(
                        "arrs can only be vector or matrix",
                    ));
                }
            }

            if iter.peek().is_some() {
                write!(w, "   ")?; // 3 spaces
            }
        }

        writeln!(w)?;
    }
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

    trait FrameNewExample {
        fn new_example() -> Frame;
    }

    impl FrameNewExample for Frame {
        fn new_example() -> Frame {
            let inp = r#"4
key1=a key2=a/b key3=a@b key4="a@b" 
Mg        -4.25650        3.79180       -2.54123
C         -1.15405        2.86652       -1.26699
C         -5.53758        3.70936        0.63504
C         -7.28250        4.71303       -3.82016
"#;
            let mut rd = Cursor::new(inp.as_bytes());
            read_frame(&mut rd).unwrap()
        }
    }

    #[test]
    fn test_write_frame() {
        // this is a round trip from a text -> Frame -> text
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
