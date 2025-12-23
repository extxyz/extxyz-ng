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
    io::{self, BufRead, Write},
    ops::Deref,
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

impl std::fmt::Display for CextxyzError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CextxyzError::Io(error) => write!(f, "{error}"),
            CextxyzError::InvalidValue(err) => write!(f, "{err}"),
        }
    }
}

impl std::error::Error for CextxyzError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            CextxyzError::Io(error) => Some(error),
            CextxyzError::InvalidValue(_) => None,
        }
    }
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

/// A newtype wrapper around `i32` that dereferences to `i32`.
///
/// # Deref coercion
///
/// `Integer` implements `Deref<Target = i32>`, allowing `&Integer` to be used
/// wherever `&i32` is expected.
///
/// ```
/// use extxyz_sys::Integer;
///
/// fn takes_i32(x: &i32) {}
///
/// let n = Integer::from(42);
/// takes_i32(&n);
/// ```
#[derive(Debug, Default, Copy, Clone)]
pub struct Integer(i32);

/// A newtype wrapper around `f64` that dereferences to `f64`.
///
/// # Deref coercion
///
/// `FloatNum` implements `Deref<Target = f64>`, allowing `&FloatNum` to be used
/// wherever `&f64` is expected.
///
/// ```
/// use extxyz_sys::FloatNum;
///
/// fn takes_f64(x: &f64) {}
///
/// let x = FloatNum::from(3.14);
/// takes_f64(&x);
/// ```
#[derive(Debug, Default, Copy, Clone)]
pub struct FloatNum(f64);

/// A newtype wrapper around `bool` that dereferences to `bool`.
///
/// # Deref coercion
///
/// `Boolean` implements `Deref<Target = bool>`, allowing `&Boolean` to be used
/// wherever `&bool` is expected.
///
/// ```
/// use extxyz_sys::Boolean;
///
/// fn takes_bool(x: &bool) {}
///
/// let b = Boolean::from(true);
/// takes_bool(&b);
/// ```
#[derive(Debug, Default, Copy, Clone)]
pub struct Boolean(bool);

/// A newtype wrapper around `String` that dereferences to `str`.
///
/// # Deref coercion
///
/// `Text` implements `Deref<Target = str>`, allowing `&Text` to be used
/// wherever `&str` is expected.
///
/// ```
/// use extxyz_sys::Text;
///
/// fn takes_str(s: &str) {}
///
/// let t = Text::from("hello");
/// takes_str(&t);
/// ```
#[derive(Debug, Default)]
pub struct Text(String);

impl Deref for Integer {
    type Target = i32;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl Deref for FloatNum {
    type Target = f64;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl Deref for Boolean {
    type Target = bool;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl Deref for Text {
    type Target = str;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl From<i32> for Integer {
    fn from(value: i32) -> Self {
        Self(value)
    }
}
impl From<f64> for FloatNum {
    fn from(value: f64) -> Self {
        Self(value)
    }
}
impl From<bool> for Boolean {
    fn from(value: bool) -> Self {
        Self(value)
    }
}
impl From<String> for Text {
    fn from(value: String) -> Self {
        Self(value)
    }
}
impl From<&str> for Text {
    fn from(value: &str) -> Self {
        Self(value.to_string())
    }
}

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
    VecInteger(Vec<Integer>, u32),
    VecFloat(Vec<FloatNum>, u32),
    VecBool(Vec<Boolean>, u32),
    VecText(Vec<Text>, u32),
    MatrixInteger(Vec<Vec<Integer>>, (u32, u32)),
    MatrixFloat(Vec<Vec<FloatNum>>, (u32, u32)),
    MatrixBool(Vec<Vec<Boolean>>, (u32, u32)),
    MatrixText(Vec<Vec<Text>>, (u32, u32)),
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
            Value::VecInteger(arr, _) => write!(f, "[{}]", fmt_array(arr)),
            Value::VecFloat(arr, _) => write!(f, "[{}]", fmt_array(arr)),
            Value::VecBool(arr, _) => write!(f, "[{}]", fmt_array(arr)),
            Value::VecText(arr, _) => write!(f, "[{}]", fmt_array(arr)),
            Value::MatrixInteger(matrix, _) => write!(f, "[{}]", fmt_matrix(matrix)),
            Value::MatrixFloat(matrix, _) => write!(f, "[{}]", fmt_matrix(matrix)),
            Value::MatrixBool(matrix, _) => write!(f, "[{}]", fmt_matrix(matrix)),
            Value::MatrixText(matrix, _) => write!(f, "[{}]", fmt_matrix(matrix)),
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
                    let vec = slice.iter().map(|i| Integer(*i)).collect::<Vec<Integer>>();
                    Value::VecInteger(vec, u32::try_from(ncols).expect("ncols out of u32 bound"))
                } else {
                    let mut matrix = Vec::with_capacity(nrows);
                    for r in 0..nrows {
                        let vec = slice[r * ncols..(r + 1) * ncols]
                            .iter()
                            .map(|i| Integer(*i))
                            .collect::<Vec<Integer>>();
                        matrix.push(vec);
                    }
                    Value::MatrixInteger(
                        matrix,
                        (
                            u32::try_from(nrows).expect("nrows out of u32 bound"),
                            u32::try_from(ncols).expect("ncols out of u32 bound"),
                        ),
                    )
                }
            }

            data_type_data_f => {
                let slice =
                    unsafe { slice::from_raw_parts(entry.data as *const f64, nrows * ncols) };

                if nrows == 1 && ncols == 1 {
                    Value::Float(FloatNum(slice[0]))
                } else if nrows == 1 {
                    let vec = slice
                        .iter()
                        .map(|i| FloatNum(*i))
                        .collect::<Vec<FloatNum>>();
                    Value::VecFloat(vec, u32::try_from(ncols).expect("ncols out of u32 bound"))
                } else {
                    let mut matrix = Vec::with_capacity(nrows);
                    for r in 0..nrows {
                        let vec = slice[r * ncols..(r + 1) * ncols]
                            .iter()
                            .map(|i| FloatNum(*i))
                            .collect::<Vec<FloatNum>>();
                        matrix.push(vec);
                    }
                    Value::MatrixFloat(
                        matrix,
                        (
                            u32::try_from(nrows).expect("nrows out of u32 bound"),
                            u32::try_from(ncols).expect("ncols out of u32 bound"),
                        ),
                    )
                }
            }

            data_type_data_b => {
                let slice =
                    unsafe { slice::from_raw_parts(entry.data as *const i32, nrows * ncols) };

                if nrows == 1 && ncols == 1 {
                    Value::Bool(Boolean(slice[0] != 0))
                } else if nrows == 1 {
                    let vec = slice
                        .iter()
                        .map(|i| Boolean(*i != 0))
                        .collect::<Vec<Boolean>>();
                    Value::VecBool(vec, u32::try_from(ncols).expect("ncols out of u32 bound"))
                } else {
                    let mut matrix = Vec::with_capacity(nrows);
                    for r in 0..nrows {
                        let vec = slice[r * ncols..(r + 1) * ncols]
                            .iter()
                            .map(|i| Boolean(*i != 0))
                            .collect::<Vec<Boolean>>();
                        matrix.push(vec);
                    }
                    Value::MatrixBool(
                        matrix,
                        (
                            u32::try_from(nrows).expect("nrows out of u32 bound"),
                            u32::try_from(ncols).expect("ncols out of u32 bound"),
                        ),
                    )
                }
            }

            data_type_data_s => {
                let slice =
                    unsafe { slice::from_raw_parts(entry.data as *const *const i8, nrows * ncols) };

                if nrows == 1 && ncols == 1 {
                    let s = unsafe { CStr::from_ptr(slice[0]).to_string_lossy().into_owned() };
                    Value::Str(Text(s))
                } else if nrows == 1 {
                    let vec = slice
                        .iter()
                        .map(|&ptr| {
                            let s = unsafe { CStr::from_ptr(ptr).to_string_lossy().into_owned() };
                            Text(s)
                        })
                        .collect::<Vec<Text>>();
                    Value::VecText(vec, u32::try_from(ncols).expect("ncols out of u32 bound"))
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
                    Value::MatrixText(
                        matrix,
                        (
                            u32::try_from(nrows).expect("nrows out of u32 bound"),
                            u32::try_from(ncols).expect("ncols out of u32 bound"),
                        ),
                    )
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

impl<'a> DictHandler {
    /// return an iter of `&(String, Value)`
    pub fn iter(&'a self) -> std::slice::Iter<'a, (String, Value)> {
        self.into_iter()
    }
}

impl<'a> IntoIterator for &'a DictHandler {
    type Item = &'a (String, Value);
    type IntoIter = std::slice::Iter<'a, (String, Value)>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
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
pub fn read_frame<R: BufRead>(
    rd: &mut R,
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

fn write_lattice<T, W>(w: &mut W, m: &[Vec<T>]) -> Result<()>
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
pub fn write_frame<W: Write>(
    w: &mut W,
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
            let i = i as usize;

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
    use std::io::{BufWriter, Cursor};

    use super::*;

    // helpers only for test
    impl Value {
        fn eq_approx(&self, other: &Value) -> bool {
            // hardcode tol = 1e-5
            let tol = 1e-5;
            match (self, other) {
                (Value::Integer(a), Value::Integer(b)) => a.0.abs_diff(b.0) == 0,
                (Value::MatrixFloat(a, a_shape), Value::MatrixFloat(b, b_shape)) => {
                    if a_shape != b_shape {
                        return false;
                    }
                    for (ax, bx) in a.iter().zip(b.iter()) {
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
        let mut rd = Cursor::new(inp.as_bytes());
        let (natoms, info, arrs) = read_frame(&mut rd, None).unwrap();

        let mut buf = Vec::new();
        {
            let mut writer = BufWriter::new(&mut buf);
            assert!(write_frame(&mut writer, natoms, &info, &arrs).is_ok());
            writer.flush().unwrap();
        }

        let mut cur = Cursor::new(&buf[..]);
        let (natoms, info, arrs) = read_frame(&mut cur, None).unwrap();

        assert_eq!(natoms, 4);
        assert_eq!(format!("{}", info.get("key1").unwrap()), "a");
        assert_eq!(format!("{}", info.get("key2").unwrap()), "a/b");
        assert_eq!(format!("{}", info.get("key3").unwrap()), "a@b");
        assert_eq!(format!("{}", info.get("key4").unwrap()), "a@b");
        assert_eq!(format!("{}", arrs.get("species").unwrap()), "[Mg, C, C, C]");

        let pos_got = arrs.get("pos").unwrap();
        let pos_expect = Value::MatrixFloat(
            Vec::from([
                Vec::from([FloatNum(-4.25650), FloatNum(3.79180), FloatNum(-2.54123)]),
                Vec::from([FloatNum(-1.15405), FloatNum(2.86652), FloatNum(-1.26699)]),
                Vec::from([FloatNum(-5.53758), FloatNum(3.70936), FloatNum(0.63504)]),
                Vec::from([FloatNum(-7.28250), FloatNum(4.71303), FloatNum(-3.82016)]),
            ]),
            (4, 3),
        );
        assert!(pos_got.eq_approx(&pos_expect));
    }
}
