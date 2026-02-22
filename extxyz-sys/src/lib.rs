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
    ffi::{CStr, CString},
    io::{self, BufRead},
    slice,
};

use extxyz_types::{Boolean, DictHandler, FloatNum, Integer, Text, Value};
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
                    Value::Integer(Integer::from(slice[0]))
                } else if nrows == 1 {
                    let vec = slice
                        .iter()
                        .map(|i| Integer::from(*i))
                        .collect::<Vec<Integer>>();
                    Value::VecInteger(vec, u32::try_from(ncols).expect("ncols out of u32 bound"))
                } else {
                    let mut matrix = Vec::with_capacity(nrows);
                    for r in 0..nrows {
                        let vec = slice[r * ncols..(r + 1) * ncols]
                            .iter()
                            .map(|i| Integer::from(*i))
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
                    Value::Float(FloatNum::from(slice[0]))
                } else if nrows == 1 {
                    let vec = slice
                        .iter()
                        .map(|i| FloatNum::from(*i))
                        .collect::<Vec<FloatNum>>();
                    Value::VecFloat(vec, u32::try_from(ncols).expect("ncols out of u32 bound"))
                } else {
                    let mut matrix = Vec::with_capacity(nrows);
                    for r in 0..nrows {
                        let vec = slice[r * ncols..(r + 1) * ncols]
                            .iter()
                            .map(|i| FloatNum::from(*i))
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
                    Value::Bool(Boolean::from(slice[0] != 0))
                } else if nrows == 1 {
                    let vec = slice
                        .iter()
                        .map(|i| Boolean::from(*i != 0))
                        .collect::<Vec<Boolean>>();
                    Value::VecBool(vec, u32::try_from(ncols).expect("ncols out of u32 bound"))
                } else {
                    let mut matrix = Vec::with_capacity(nrows);
                    for r in 0..nrows {
                        let vec = slice[r * ncols..(r + 1) * ncols]
                            .iter()
                            .map(|i| Boolean::from(*i != 0))
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
                    Value::Str(Text::from(s))
                } else if nrows == 1 {
                    let vec = slice
                        .iter()
                        .map(|&ptr| {
                            let s = unsafe { CStr::from_ptr(ptr).to_string_lossy().into_owned() };
                            Text::from(s)
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
                                Text::from(s)
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

trait FromPtr {
    unsafe fn from_ptr(ptr: *mut dict_entry_struct) -> Self;
}

impl FromPtr for DictHandler {
    /// Create an owned dict from ptr.
    ///
    /// # Safety
    ///
    /// public function might dereference a raw pointer but is not marked `unsafe`.
    /// Make sure the raw ptr is valid.
    unsafe fn from_ptr(ptr: *mut dict_entry_struct) -> Self {
        let data = c_to_rust_dict(ptr);
        DictHandler(data)
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
    let (info_val, arrays_val) =
        unsafe { (DictHandler::from_ptr(info), DictHandler::from_ptr(arrays)) };

    unsafe {
        free_dict(info);
        free_dict(arrays);
    }

    Ok((nat as u32, info_val, arrays_val))
}
