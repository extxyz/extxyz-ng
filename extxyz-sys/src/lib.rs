#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(unnecessary_transmutes)]

use std::io;

include!("bindings.rs");

#[derive(Debug)]
struct Dict {
    inner: DictEntry,
}

/// Safe wrapper of unsafe c api for `extxyz_read_ll` function
/// It returns, (natoms, info, arrays, comments) as a fallible result
fn extxyz_read(input: &str) -> Result<(i32, DictEntry, DictEntry, String), io::Error> {
    let c_input = std::ffi::CString::new(input)
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "Input contains null byte"))?;

    let kv_grammar = unsafe { compile_extxyz_kv_grammar() };

    // Prepare output variables
    let mut nat: i32 = 0;
    // allocate buffer for comment
    let mut comment_buf = vec![0u8; 1024]; // choose suitable length
    let comment_ptr = comment_buf.as_mut_ptr().cast::<i8>();

    // allocate pointer for info ptr and arrays ptr
    let mut info: *mut DictEntry = std::ptr::null_mut();
    let mut arrays: *mut DictEntry = std::ptr::null_mut();

    let ret = unsafe {
        let fp: *mut FILE = libc::fopen(c_input.as_ptr(), c"r".as_ptr().cast::<i8>());
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
            std::ptr::null_mut(), // error message pointer
        )
    };

    if ret != 0 {
        return Err(io::Error::other("extxyz_read_ll failed"));
    }

    // convert comment buffer to Rust String
    let comment = unsafe {
        std::ffi::CStr::from_ptr(comment_ptr)
            .to_string_lossy()
            .into_owned()
    };

    // XXX: can I make sure c side handle the clear? I think no.
    // I should call free_dict explicitly on wrap it in a Drop
    let info_val = unsafe { *info };
    let arrays_val = unsafe { *arrays };

    Ok((nat, info_val, arrays_val, comment))
}
