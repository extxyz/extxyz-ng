#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(unnecessary_transmutes)]
#![allow(clippy::if_not_else)]

use std::{
    collections::HashMap,
    ffi::{CStr, CString},
    io, slice,
};

use libc::fmemopen;

include!("bindings.rs");

#[derive(Debug)]
pub enum Value {
    Int(i32),
    Float(f64),
    Bool(bool),
    Str(String),
    IntArray(Vec<i32>),
    FloatArray(Vec<f64>),
    BoolArray(Vec<bool>),
    StrArray(Vec<String>),
    MatrixInt(Vec<Vec<i32>>),
    MatrixFloat(Vec<Vec<f64>>),
    MatrixBool(Vec<Vec<bool>>),
    MatrixStr(Vec<Vec<String>>),
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
            Value::Int(v) => write!(f, "{v}"),
            Value::Float(v) => write!(f, "{v}"),
            Value::Bool(v) => write!(f, "{v}"),
            Value::Str(s) => write!(f, "{s}"),
            Value::IntArray(arr) => write!(f, "[{}]", fmt_array(arr)),
            Value::FloatArray(arr) => write!(f, "[{}]", fmt_array(arr)),
            Value::BoolArray(arr) => write!(f, "[{}]", fmt_array(arr)),
            Value::StrArray(arr) => write!(f, "[{}]", fmt_array(arr)),
            Value::MatrixInt(matrix) => write!(f, "[{}]", fmt_matrix(matrix)),
            Value::MatrixFloat(matrix) => write!(f, "[{}]", fmt_matrix(matrix)),
            Value::MatrixBool(matrix) => write!(f, "[{}]", fmt_matrix(matrix)),
            Value::MatrixStr(matrix) => write!(f, "[{}]", fmt_matrix(matrix)),
            Value::Unsupported => write!(f, "<unsupported>"),
        }
    }
}

#[allow(clippy::cast_sign_loss)]
unsafe fn c_to_rust_dict(mut ptr: *mut dict_entry_struct) -> HashMap<String, Value> {
    let mut map = HashMap::new();

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
                    Value::Int(slice[0])
                } else if nrows == 1 {
                    Value::IntArray(slice.to_vec())
                } else {
                    let mut matrix = Vec::with_capacity(nrows);
                    for r in 0..nrows {
                        matrix.push(slice[r * ncols..(r + 1) * ncols].to_vec());
                    }
                    Value::MatrixInt(matrix)
                }
            }

            data_type_data_f => {
                let slice =
                    unsafe { slice::from_raw_parts(entry.data as *const f64, nrows * ncols) };
                if nrows == 1 && ncols == 1 {
                    Value::Float(slice[0])
                } else if nrows == 1 {
                    Value::FloatArray(slice.to_vec())
                } else {
                    let mut matrix = Vec::with_capacity(nrows);
                    for r in 0..nrows {
                        matrix.push(slice[r * ncols..(r + 1) * ncols].to_vec());
                    }
                    Value::MatrixFloat(matrix)
                }
            }

            data_type_data_b => {
                let slice =
                    unsafe { slice::from_raw_parts(entry.data as *const i32, nrows * ncols) };
                if nrows == 1 && ncols == 1 {
                    Value::Bool(slice[0] != 0)
                } else if nrows == 1 {
                    Value::BoolArray(slice.iter().map(|&v| v != 0).collect())
                } else {
                    let mut matrix = Vec::with_capacity(nrows);
                    for r in 0..nrows {
                        matrix.push(
                            slice[r * ncols..(r + 1) * ncols]
                                .iter()
                                .map(|&v| v != 0)
                                .collect(),
                        );
                    }
                    Value::MatrixBool(matrix)
                }
            }

            data_type_data_s => {
                let slice =
                    unsafe { slice::from_raw_parts(entry.data as *const *const i8, nrows * ncols) };
                if nrows == 1 && ncols == 1 {
                    let s = unsafe { CStr::from_ptr(slice[0]).to_string_lossy().into_owned() };
                    Value::Str(s)
                } else if nrows == 1 {
                    let vec: Vec<String> = slice
                        .iter()
                        .map(|&p| unsafe { CStr::from_ptr(p).to_string_lossy().into_owned() })
                        .collect();
                    Value::StrArray(vec)
                } else {
                    let mut matrix = Vec::with_capacity(nrows);
                    for r in 0..nrows {
                        let row = slice[r * ncols..(r + 1) * ncols]
                            .iter()
                            .map(|&p| unsafe { CStr::from_ptr(p).to_string_lossy().into_owned() })
                            .collect();
                        matrix.push(row);
                    }
                    Value::MatrixStr(matrix)
                }
            }

            _ => Value::Unsupported,
        };

        map.insert(key, value);

        ptr = entry.next;
    }

    map
}

// Safe hardler for `DictEntry`
#[derive(Debug)]
pub struct DictHandler(HashMap<String, Value>);

impl DictHandler {
    /// Create an owned dict from ptr.
    ///
    /// # Safety
    ///
    /// public function might dereference a raw pointer but is not marked `unsafe`.
    /// Make sure the raw ptr is valid.
    pub unsafe fn new(ptr: *mut dict_entry_struct) -> Self {
        let data = unsafe { c_to_rust_dict(ptr) };
        DictHandler(data)
    }

    #[must_use]
    pub fn get(&self, key: &str) -> Option<&Value> {
        self.0.get(key)
    }
}

/// Safe wrapper around the unsafe C API function `extxyz_read_ll`.
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
pub fn extxyz_read(
    input: &str,
    comment_override: Option<&str>,
) -> Result<(i32, DictHandler, DictHandler), io::Error> {
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

    let ret = unsafe {
        let mut bytes = input.as_bytes().to_vec();
        let fp = fmemopen(
            bytes.as_mut_ptr().cast::<libc::c_void>(),
            bytes.len(),
            CString::new("r")
                .expect("cannot have internal 0 byte")
                .as_ptr(),
        );
        if fp.is_null() {
            return Err(io::Error::other("Failed to open file"));
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

    let err_msg = unsafe {
        let err_cstr = CStr::from_ptr(error_ptr.cast_const());
        err_cstr.to_string_lossy()
    };

    // NOTE: extxyz use 1 for success 0 for failed state code, fuckers
    if ret != 1 {
        return Err(io::Error::other(format!(
            "extxyz_read_ll failed with msg: {err_msg}",
        )));
    }

    // own the dict and it will be dropped after use.
    let (info_val, arrays_val) = unsafe { (DictHandler::new(info), DictHandler::new(arrays)) };

    unsafe {
        free_dict(info);
        free_dict(arrays);
    }

    Ok((nat, info_val, arrays_val))
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn extxyz_read_default() {
        let inp = r#"4
key1=a key2=a/b key3=a@b key4="a@b"
Mg        -4.25650        3.79180       -2.54123
C         -1.15405        2.86652       -1.26699
C         -5.53758        3.70936        0.63504
C         -7.28250        4.71303       -3.82016
"#;
        let (natoms, info, arr) = extxyz_read(inp, None).unwrap();

        assert_eq!(natoms, 4);
        assert_eq!(format!("{}", info.get("key1").unwrap()), "a");
        assert_eq!(format!("{}", info.get("key2").unwrap()), "a/b");
        assert_eq!(format!("{}", info.get("key3").unwrap()), "a@b");
        assert_eq!(format!("{}", info.get("key4").unwrap()), "a@b");
        assert_eq!(format!("{}", arr.get("species").unwrap()), "[Mg, C, C, C]");
        assert_eq!(format!("{}", arr.get("pos").unwrap()), "[[-4.2565, 3.7918, -2.54123], [-1.15405, 2.86652, -1.26699], [-5.53758, 3.70936, 0.63504], [-7.2825, 4.71303, -3.82016]]");
    }
}
