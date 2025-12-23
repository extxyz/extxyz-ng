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
    io::{self, BufRead},
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
#[must_use]
pub fn escape(s: &str) -> Cow<'_, str> {
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
pub struct DictHandler(pub Vec<(String, Value)>);

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
