#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(unnecessary_transmutes)]
#![allow(clippy::if_not_else)]
#![allow(clippy::too_many_lines, clippy::useless_format)]
#![allow(clippy::match_bool)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::needless_range_loop)]

use std::{
    borrow::Cow,
    ffi::{CStr, CString},
    io::{self, BufReader, BufWriter, Read, Write},
    slice,
};

use libc::fmemopen;

include!("bindings.rs");

pub type Result<T> = std::result::Result<T, CextxyzError>;

#[derive(Debug)]
pub enum CextxyzError {
    Io(std::io::Error),
    InvalidValue(&'static str),
}

impl From<std::io::Error> for CextxyzError {
    fn from(value: std::io::Error) -> Self {
        CextxyzError::Io(value)
    }
}

/// checking special characters escape and escape as needed, using Cow because most string won't
/// need quoting.
fn escape(s: &str) -> Cow<'_, str> {
    let needs_quoting = s.chars().any(|c| {
        matches!(
            c,
            '"' | '\\' | '\n' | ' ' | '=' | ',' | '[' | ']' | '{' | '}'
        )
    });

    if !needs_quoting {
        return Cow::Borrowed(s);
    }

    // +4 is a fair guess for capacity with x2 quotes and possibly 2 escapes
    let mut out = String::with_capacity(s.len() + 4);
    out.push('"');
    for c in s.chars() {
        match c {
            '\n' => out.push_str("\\n"),
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            _ => out.push(c),
        }
    }
    out.push('"');

    Cow::Owned(out)
}

#[derive(Debug, Default, Copy, Clone)]
pub struct Integer(i32);
#[derive(Debug, Default, Copy, Clone)]
pub struct FloatNum(f64);
#[derive(Debug, Default, Copy, Clone)]
pub struct Boolean(bool);
#[derive(Debug, Default)]
pub struct Text(String);

// In the c impl the output format to:
// #define INTEGER_FMT "%8d"
// #define FLOAT_FMT "%16.8f"
// #define STRING_FMT "%s"
// #define BOOL_FMT "%.1s"

impl std::fmt::Display for Integer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:8}", self.0)
    }
}
impl std::fmt::Display for FloatNum {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:16.8}", self.0)
    }
}
impl std::fmt::Display for Boolean {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.0 {
            true => write!(f, "T"),
            false => write!(f, "F"),
        }
    }
}
impl std::fmt::Display for Text {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", escape(&self.0))
    }
}

#[derive(Debug)]
pub enum Value {
    Integer(Integer),
    Float(FloatNum),
    Bool(Boolean),
    Str(Text),
    VecInteger(Vec<Integer>),
    VecFloat(Vec<FloatNum>),
    VecBool(Vec<Boolean>),
    VecText(Vec<Text>),
    MatrixInteger(Vec<Vec<Integer>>),
    MatrixFloat(Vec<Vec<FloatNum>>),
    MatrixBool(Vec<Vec<Boolean>>),
    MatrixText(Vec<Vec<Text>>),
    Unsupported,
}

impl std::fmt::Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        fn fmt_array<T: std::fmt::Display>(arr: &[T]) -> String {
            arr.iter()
                .map(std::string::ToString::to_string)
                .collect::<Vec<_>>()
                .join(", ")
        }

        fn fmt_matrix<T: std::fmt::Display>(matrix: &[Vec<T>]) -> String {
            matrix
                .iter()
                .map(|row| format!("[{}]", fmt_array(row)))
                .collect::<Vec<_>>()
                .join(", ")
        }

        match self {
            Value::Integer(v) => write!(f, "{v}"),
            Value::Float(v) => write!(f, "{v}"),
            Value::Bool(v) => write!(f, "{v}"),
            Value::Str(v) => write!(f, "{v}"),
            Value::VecInteger(arr) => write!(f, "[{}]", fmt_array(arr)),
            Value::VecFloat(arr) => write!(f, "[{}]", fmt_array(arr)),
            Value::VecBool(arr) => write!(f, "[{}]", fmt_array(arr)),
            Value::VecText(arr) => write!(f, "[{}]", fmt_array(arr)),
            Value::MatrixInteger(matrix) => write!(f, "[{}]", fmt_matrix(matrix)),
            Value::MatrixFloat(matrix) => write!(f, "[{}]", fmt_matrix(matrix)),
            Value::MatrixBool(matrix) => write!(f, "[{}]", fmt_matrix(matrix)),
            Value::MatrixText(matrix) => write!(f, "[{}]", fmt_matrix(matrix)),
            Value::Unsupported => write!(f, "<unsupported>"),
        }
    }
}

unsafe fn c_to_rust_dict(mut ptr: *mut dict_entry_struct) -> Vec<(String, Value)> {
    let mut map = Vec::new();

    while !ptr.is_null() {
        let entry = unsafe { &*ptr };

        // Convert key
        let key = if !entry.key.is_null() {
            unsafe { CStr::from_ptr(entry.key).to_string_lossy().into_owned() }
        } else {
            panic!("Key cannot be null");
        };

        // Ensure at least 1 row/col for compatibility with C loops
        let nrows = if entry.nrows < 1 {
            1
        } else {
            entry.nrows as usize
        };
        let ncols = if entry.ncols < 1 {
            1
        } else {
            entry.ncols as usize
        };

        let value = match entry.data_t {
            data_type_data_i => {
                let slice =
                    unsafe { slice::from_raw_parts(entry.data as *const i32, nrows * ncols) };

                if nrows == 1 && ncols == 1 {
                    Value::Integer(Integer(slice[0]))
                } else if nrows == 1 {
                    let vecint = slice.iter().map(|i| Integer(*i)).collect::<Vec<Integer>>();
                    Value::VecInteger(vecint)
                } else {
                    let mut matrix = Vec::with_capacity(nrows);
                    for r in 0..nrows {
                        let vecint = slice[r * ncols..(r + 1) * ncols]
                            .iter()
                            .map(|i| Integer(*i))
                            .collect::<Vec<Integer>>();
                        matrix.push(vecint);
                    }
                    Value::MatrixInteger(matrix)
                }
            }

            data_type_data_f => {
                let slice =
                    unsafe { slice::from_raw_parts(entry.data as *const f64, nrows * ncols) };

                if nrows == 1 && ncols == 1 {
                    Value::Float(FloatNum(slice[0]))
                } else if nrows == 1 {
                    let vecint = slice
                        .iter()
                        .map(|i| FloatNum(*i))
                        .collect::<Vec<FloatNum>>();
                    Value::VecFloat(vecint)
                } else {
                    let mut matrix = Vec::with_capacity(nrows);
                    for r in 0..nrows {
                        let vecint = slice[r * ncols..(r + 1) * ncols]
                            .iter()
                            .map(|i| FloatNum(*i))
                            .collect::<Vec<FloatNum>>();
                        matrix.push(vecint);
                    }
                    Value::MatrixFloat(matrix)
                }
            }

            data_type_data_b => {
                let slice =
                    unsafe { slice::from_raw_parts(entry.data as *const i32, nrows * ncols) };

                if nrows == 1 && ncols == 1 {
                    Value::Bool(Boolean(slice[0] != 0))
                } else if nrows == 1 {
                    let vecint = slice
                        .iter()
                        .map(|i| Boolean(*i != 0))
                        .collect::<Vec<Boolean>>();
                    Value::VecBool(vecint)
                } else {
                    let mut matrix = Vec::with_capacity(nrows);
                    for r in 0..nrows {
                        let vecint = slice[r * ncols..(r + 1) * ncols]
                            .iter()
                            .map(|i| Boolean(*i != 0))
                            .collect::<Vec<Boolean>>();
                        matrix.push(vecint);
                    }
                    Value::MatrixBool(matrix)
                }
            }

            data_type_data_s => {
                let slice =
                    unsafe { slice::from_raw_parts(entry.data as *const *const i8, nrows * ncols) };

                if nrows == 1 && ncols == 1 {
                    let s = unsafe { CStr::from_ptr(slice[0]).to_string_lossy().into_owned() };
                    Value::Str(Text(s))
                } else if nrows == 1 {
                    let vecint = slice
                        .iter()
                        .map(|&ptr| {
                            let s = unsafe { CStr::from_ptr(ptr).to_string_lossy().into_owned() };
                            Text(s)
                        })
                        .collect::<Vec<Text>>();
                    Value::VecText(vecint)
                } else {
                    let mut matrix = Vec::with_capacity(nrows);
                    for r in 0..nrows {
                        let vecint = slice[r * ncols..(r + 1) * ncols]
                            .iter()
                            .map(|&ptr| {
                                let s =
                                    unsafe { CStr::from_ptr(ptr).to_string_lossy().into_owned() };
                                Text(s)
                            })
                            .collect::<Vec<Text>>();
                        matrix.push(vecint);
                    }
                    Value::MatrixText(matrix)
                }
            }

            _ => Value::Unsupported,
        };

        map.push((key, value));

        ptr = entry.next;
    }

    map
}

// Safe hardler for `DictEntry`
#[derive(Debug)]
pub struct DictHandler(Vec<(String, Value)>);

impl DictHandler {
    /// Create an owned dict from ptr.
    ///
    /// # Safety
    ///
    /// public function might dereference a raw pointer but is not marked `unsafe`.
    /// Make sure the raw ptr is valid.
    unsafe fn new(ptr: *mut dict_entry_struct) -> Self {
        let data = c_to_rust_dict(ptr);
        DictHandler(data)
    }

    /// Get the value by key.
    /// Since internally extxyz dict stores not as a real hashmap but a linklist,
    /// and the lookup takes O(N).
    #[must_use]
    pub fn get(&self, key: &str) -> Option<&Value> {
        for (k, v) in &self.0 {
            if k.as_str() == key {
                return Some(v);
            }
        }

        None
    }
}

/// Safe wrapper around the unsafe C API function `extxyz_read_ll`.
///
/// TODO: `extxyz_read_ll` takes fp, which has an internal c buffer, the wrapper accept &str as input
/// is too specific and lost performance when reading large files.
///
/// Returns `(natoms, info, arrays, comments)` as a fallible `Result`.
///
/// # Errors
///
/// - If the input contains a null byte.
/// - If reading the input inside the unsafe block fails.
/// - If the unsafe `extxyz_read_ll` call returns an error.
///
/// # Panics
///
/// - If initialization of a `CString` fails.
///
/// XXX: the general wrapper takes *mut FILE as argument, and can then have
/// - `extxyz_read_from_file` and
/// - `extxyz_read_from_str`.
pub fn frame_read<T: Read>(
    mut rd: BufReader<T>,
    comment_override: Option<&str>,
) -> Result<(u32, DictHandler, DictHandler)> {
    let kv_grammar = unsafe { compile_extxyz_kv_grammar() };

    // Prepare output variables
    let mut nat: i32 = 0;

    let comment_ptr = match comment_override {
        Some(_) => {
            let mut comment_buf = vec![0; 1024];
            comment_buf.as_mut_ptr().cast::<i8>()
        }
        None => std::ptr::null_mut(),
    };

    // allocate pointer for info ptr and arrays ptr
    // NOTE: the info and arrays are ptr and they are allocated within the `extxyz_read_ll`
    // unsafe call through `tree_to_dict`
    let mut info: *mut DictEntry = std::ptr::null_mut();
    let mut arrays: *mut DictEntry = std::ptr::null_mut();

    let mut error_message = vec![0u8; 1024];
    let error_ptr = error_message.as_mut_ptr().cast::<i8>();

    let mut buf = Vec::new();
    rd.read_to_end(&mut buf)?;
    let mut bytes = buf;
    let fp = unsafe {
        fmemopen(
            bytes.as_mut_ptr().cast::<libc::c_void>(),
            bytes.len(),
            CString::new("r")
                .expect("cannot have internal 0 byte")
                .as_ptr(),
        )
    };

    let ret = unsafe {
        if fp.is_null() {
            return Err(io::Error::other("Failed to open file").into());
        }

        extxyz_read_ll(
            kv_grammar,
            fp,
            &raw mut nat,
            &raw mut info,
            &raw mut arrays,
            comment_ptr,
            error_ptr,
        )
    };

    unsafe { libc::fclose(fp) };

    let err_msg = unsafe {
        let err_cstr = CStr::from_ptr(error_ptr.cast_const());
        err_cstr.to_string_lossy()
    };

    // NOTE: extxyz use 1 for success 0 for failed state code, fuckers
    if ret != 1 {
        return Err(io::Error::other(format!("extxyz_read_ll failed with msg: {err_msg}",)).into());
    }

    // own the dict and it will be dropped after use.
    let (info_val, arrays_val) = unsafe { (DictHandler::new(info), DictHandler::new(arrays)) };

    unsafe {
        free_dict(info);
        free_dict(arrays);
    }

    Ok((nat as u32, info_val, arrays_val))
}

fn write_lattice<T, W>(w: &mut BufWriter<W>, m: &[Vec<T>]) -> Result<()>
where
    T: Default + std::fmt::Display + Copy,
    W: Write,
{
    if m.len() != 3 {
        return Err(CextxyzError::InvalidValue("expect 3x3 matrix"));
    }

    // transpose lattice matrix which has vectors in column-wise.
    let mut m3 = [[T::default(); 3]; 3];

    for i in 0..3 {
        if m[i].len() != 3 {
            return Err(CextxyzError::InvalidValue("expect 3x3 matrix"));
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

fn write_vec<T, W>(w: &mut BufWriter<W>, s: &[T]) -> Result<()>
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
pub fn frame_write<W: Write>(
    w: &mut BufWriter<W>,
    natoms: u32,
    info: &DictHandler,
    arrs: &DictHandler,
) -> Result<()> {
    writeln!(w, "{natoms}")?;

    // info
    let mut iter = info.0.iter().peekable();
    while let Some((k, v)) = iter.next() {
        // the inner datastructure will store "Properties" as a key (if exist), but in the
        // write function the Properties field is deduct from the arr.
        // When read the xyz may not have "Properties" field, but write will always have it.
        if k.as_str() == "Properties" {
            continue;
        }

        let s = escape(k);
        write!(w, "{s}")?;
        write!(w, "=")?;

        // in extxyz c implementation, lattice treated different write in column-wise and use
        // single space as spliter
        if k.as_str() == "Lattice" {
            match v {
                Value::MatrixInteger(m) => {
                    write_lattice(w, m)?;
                }
                Value::MatrixFloat(m) => {
                    write_lattice(w, m)?;
                }
                Value::MatrixBool(m) => {
                    write_lattice(w, m)?;
                }
                _ => {
                    // this is unreachable if the inner dict is not create manually
                    return Err(CextxyzError::InvalidValue(
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
    let mut iter = arrs.0.iter().peekable();
    // for (k, v) in &arrs.0 {
    while let Some((k, v)) = iter.next() {
        s.push_str(k);
        s.push(':');
        match v {
            Value::VecInteger(_) => s.push_str("I:1"),
            Value::VecFloat(_) => s.push_str("R:1"),
            Value::VecBool(_) => s.push_str("L:1"),
            Value::VecText(_) => s.push_str("S:1"),
            Value::MatrixInteger(m) => s.push_str(format!("I:{}", m[0].len()).as_str()),
            Value::MatrixFloat(m) => s.push_str(format!("R:{}", m[0].len()).as_str()),
            Value::MatrixBool(m) => s.push_str(format!("L:{}", m[0].len()).as_str()),
            Value::MatrixText(m) => s.push_str(format!("S:{}", m[0].len()).as_str()),
            _ => {
                // this is unreachable if the inner dict is not create manually
                return Err(CextxyzError::InvalidValue(
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
        let mut iter = arrs.0.iter().peekable();
        while let Some((_, v)) = iter.next() {
            match v {
                Value::VecInteger(items) => write!(w, "{}", items[i as usize])?,
                Value::VecFloat(items) => write!(w, "{}", items[i as usize])?,
                Value::VecBool(items) => write!(w, "{}", items[i as usize])?,
                Value::VecText(items) => write!(w, "{}", items[i as usize])?,
                Value::MatrixInteger(items) => {
                    let s = &items[i as usize];
                    write_vec(w, s)?;
                }
                Value::MatrixFloat(items) => {
                    let s = &items[i as usize];
                    write_vec(w, s)?;
                }
                Value::MatrixBool(items) => {
                    let s = &items[i as usize];
                    write_vec(w, s)?;
                }
                Value::MatrixText(items) => {
                    let s = &items[i as usize];
                    write_vec(w, s)?;
                }
                _ => {
                    // this is unreachable if the inner dict is not create manually
                    return Err(CextxyzError::InvalidValue(
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

    Ok(())
}

#[cfg(test)]
mod test {
    use std::io::Cursor;

    use super::*;

    // helpers only for test
    impl Value {
        fn eq_approx(&self, other: &Value) -> bool {
            // hardcode tol = 1e-5
            let tol = 1e-5;
            match (self, other) {
                (Value::Integer(a), Value::Integer(b)) => a.0.abs_diff(b.0) == 0,
                (Value::MatrixFloat(a), Value::MatrixFloat(b)) => {
                    if a.len() != b.len() {
                        return false;
                    }
                    for (ax, bx) in a.iter().zip(b.iter()) {
                        if ax.len() != bx.len() {
                            return false;
                        }
                        for (v, u) in ax.iter().zip(bx.iter()) {
                            if f64::abs(v.0 - u.0) > tol {
                                return false;
                            }
                        }
                    }

                    true
                }
                _ => false,
            }
        }
    }

    // a round trip read and write
    #[test]
    fn extxyz_rw_round_trip() {
        let inp = r#"4
key1=a key2=a/b key3=a@b key4="a@b" 
Mg        -4.25650        3.79180       -2.54123
C         -1.15405        2.86652       -1.26699
C         -5.53758        3.70936        0.63504
C         -7.28250        4.71303       -3.82016
"#;
        let rd = BufReader::new(Cursor::new(inp.as_bytes()));
        let (natoms, info, arrs) = frame_read(rd, None).unwrap();

        let mut buf = Vec::new();
        {
            let mut writer = BufWriter::new(&mut buf);
            assert!(frame_write(&mut writer, natoms, &info, &arrs).is_ok());
            writer.flush().unwrap();
        }

        let rd = BufReader::new(&buf[..]);
        let (natoms, info, arrs) = frame_read(rd, None).unwrap();

        assert_eq!(natoms, 4);
        assert_eq!(format!("{}", info.get("key1").unwrap()), "a");
        assert_eq!(format!("{}", info.get("key2").unwrap()), "a/b");
        assert_eq!(format!("{}", info.get("key3").unwrap()), "a@b");
        assert_eq!(format!("{}", info.get("key4").unwrap()), "a@b");
        assert_eq!(format!("{}", arrs.get("species").unwrap()), "[Mg, C, C, C]");

        let pos_got = arrs.get("pos").unwrap();
        let pos_expect = Value::MatrixFloat(Vec::from([
            Vec::from([FloatNum(-4.25650), FloatNum(3.79180), FloatNum(-2.54123)]),
            Vec::from([FloatNum(-1.15405), FloatNum(2.86652), FloatNum(-1.26699)]),
            Vec::from([FloatNum(-5.53758), FloatNum(3.70936), FloatNum(0.63504)]),
            Vec::from([FloatNum(-7.28250), FloatNum(4.71303), FloatNum(-3.82016)]),
        ]));
        assert!(pos_got.eq_approx(&pos_expect));
    }
}
