use crate::{error::ExtxyzError, Result};
use std::io::Write;

use extxyz_types::{escape, Frame, Value};

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

/// Writes a single frame in extended XYZ (extxyz) format.
///
/// This function serializes a [`Frame`] into the extxyz text format and writes
/// it to the provided writer. 
/// The output includes a `Properties` field derived from the frame's
/// array data, even if it was not explicitly present in the input. The `Lattice`
/// field, if present, is written in column-major order following the extxyz
/// specification.
///
/// # Parameters
/// - `w`: A writer implementing [`Write`] to which the frame will be written.
/// - `frame`: The [`Frame`] to serialize.
///
/// # Errors
/// Returns an error if:
/// - Writing to the underlying writer fails.
/// - The `Lattice` field exists but is not a valid 3×3 integer or float matrix.
/// - Any internal formatting or serialization step fails.
///
/// # Notes
/// - Keys and values in the frame's metadata are escaped as required by the
///   extxyz format.
/// - The ordering of metadata fields follows the internal ordering of the frame.
/// - The `Properties` field is not taken directly from metadata but inferred
///   from the atomic data arrays.
///
/// # Examples
/// ```ignore
/// use std::fs::File;
/// use extxyz::write_frame;
//
/// let mut file = File::create("output.xyz")?;
/// write_frame(&mut file, &frame)?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn write_frame<W>(w: &mut W, frame: &Frame) -> Result<()>
where
    W: Write,
{
    let natoms = frame.natoms();

    writeln!(w, "{natoms}")?;

    // info
    let info = frame.info_orderd();
    let mut iter = info.iter().peekable();
    while let Some((k, v)) = iter.next() {
        // the inner datastructure will store "Properties" as a key (if exist), but in the
        // write function the Properties field is deduct from the arr.
        // When read the xyz may not have "Properties" field, but write will always have it.
        // XXX: therefore in the read (the parser, if I impl myself) need to validate the
        // properties is conform with what provided.
        if *k == "Properties" {
            continue;
        }

        let s = escape(k);
        write!(w, "{s}")?;
        write!(w, "=")?;

        // in extxyz c implementation, lattice treated different write in column-wise and use
        // single space as spliter
        if *k == "Lattice" {
            match v {
                Value::MatrixInteger(_, _) => {
                    write!(w, "{v}")?;
                }
                Value::MatrixFloat(_, _) => {
                    write!(w, "{v}")?;
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
    write!(w, "Properties=")?;

    let mut s = String::new();
    let arrs = frame.arrs_orderd();
    let mut iter = arrs.iter().peekable();
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

    // arrays
    for i in 0..natoms {
        let mut iter = arrs.iter().peekable();
        while let Some((_, v)) = iter.next() {
            let i = i as usize;

            // In legacy the libAtoms/extxyz's c impl the output format to:
            // #define INTEGER_FMT "%8d"
            // #define FLOAT_FMT "%16.8f"
            // #define STRING_FMT "%s"
            // #define BOOL_FMT "%.1s"

            // new format rule of arr printing is:
            // - for text, if it <8, pad to width 8 and left align (backward compatible otherwise
            // shitty libAtoms/extxyz throw segfault), if its len >8 padding len +2
            // - for float, the single value trimed to .8 precision with width of 16.
            // - interger %8d
            // - bool .1s

            // store the columns but write row by row
            match v {
                Value::VecInteger(items, _) => write!(w, "{}", items[i])?,
                Value::VecFloat(items, _) => write!(w, "{:>16.8}", items[i])?,
                Value::VecBool(items, _) => write!(w, "{}", items[i])?,
                Value::VecText(items, _) => {
                    let s = &items[i];
                    let sl = (*s).len();
                    if sl > 5 {
                        write!(w, "{1:<0$}", sl + 2, items[i])?
                    } else {
                        write!(w, "{:<5}", items[i])?
                    }
                }
                Value::MatrixInteger(items, _) => {
                    let s = &items[i];
                    let indent = " ";
                    let s = s
                        .iter()
                        .map(|i| format!("{i}"))
                        .collect::<Vec<_>>()
                        .join(indent);
                    write!(w, "{s}")?;
                }
                Value::MatrixFloat(items, _) => {
                    let s = &items[i];
                    let indent = " ";
                    let s = s
                        .iter()
                        .map(|i| {
                            let s = format!("{:>16.8}", i);
                            s
                        })
                        .collect::<Vec<_>>()
                        .join(indent);
                    write!(w, "{s}")?;
                }
                Value::MatrixBool(items, _) => {
                    let s = &items[i];
                    let indent = " ";
                    let s = s
                        .iter()
                        .map(|i| format!("{i}"))
                        .collect::<Vec<_>>()
                        .join(indent);
                    write!(w, "{s}")?;
                }
                Value::MatrixText(items, _) => {
                    let s = &items[i];
                    let indent = " ";
                    let s = s
                        .iter()
                        .map(|i| format!("{i}"))
                        .collect::<Vec<_>>()
                        .join(indent);
                    write!(w, "{s}")?;
                }
                _ => {
                    // this is unreachable if the inner dict is not create manually
                    return Err(ExtxyzError::InvalidValue(
                        "arrs can only be vector or matrix",
                    ));
                }
            }

            if iter.peek().is_some() {
                write!(w, " ")?; // 1 enforced space
            }
        }

        writeln!(w)?;
    }
    w.flush()?;

    Ok(())
}

// for multiframe writer
// #[derive(Debug)]
// pub struct FrameWriter<W> {
//     w: W,
// }
//
// impl<W: Write> FrameWriter<W> {
//     pub fn new(w: W) -> Self {
//         Self { w }
//     }
//
//     pub fn write(&mut self, frame: &Frame) -> Result<()> {
//         write_frame(&mut self.w, frame)?;
//
//         Ok(())
//     }
//
//     pub fn finish(&mut self) -> Result<()> {
//         self.w.flush()?;
//         Ok(())
//     }
// }

#[cfg(test)]
mod tests {
    use std::io::{BufWriter, Cursor};

    use crate::read_frame;

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

    // // For test printing purpose
    // struct TFrame(Frame);
    //
    // impl std::fmt::Display for TFrame {
    //     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    //         let mut buf = Vec::new();
    //         write_frame(&mut buf, &self.0).map_err(|_| std::fmt::Error)?;
    //         let s = std::str::from_utf8(&buf).map_err(|_| std::fmt::Error)?;
    //         f.write_str(s)
    //     }
    // }

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
Mg         -4.25650000       3.79180000      -2.54123000
C          -1.15405000       2.86652000      -1.26699000
C          -5.53758000       3.70936000       0.63504000
C          -7.28250000       4.71303000      -3.82016000
"#;
        assert_eq!(s, expect);
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
