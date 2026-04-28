#![allow(non_camel_case_types)]

use std::{io::BufReader, path::PathBuf};

use extxyz::{
    read_frame as rs_read_frame, write_frame as rs_write_frame, Frame, FrameReaderOwned,
    Value as InnerValue,
};
use pyo3::{
    prelude::*,
    types::{PyBool, PyDict, PyList, PyString},
};

struct Value(InnerValue);

impl<'py> IntoPyObject<'py> for Value {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    // Error can be Infallible because we know the exact type of what Value included so the
    // conversion to python type is predictable and cannot fail.
    type Error = std::convert::Infallible;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        // unwrap() is used safely because the error is infallible.
        let obj = match self.0 {
            InnerValue::Integer(i) => (*i).into_pyobject(py)?.into_any(),
            InnerValue::Float(f) => (*f).into_pyobject(py)?.into_any(),
            InnerValue::Bool(b) => PyBool::new(py, *b).to_owned().into_any(),
            InnerValue::Str(s) => (*s).into_pyobject(py)?.into_any(),

            InnerValue::VecInteger(v, _) => {
                let list = PyList::new(py, v.into_iter().map(|x| x.into_pyobject(py).unwrap()))
                    .expect("vec of int to 1d list");
                list.into_any()
            }

            InnerValue::VecFloat(v, _) => {
                let list = PyList::new(py, v.into_iter().map(|x| x.into_pyobject(py).unwrap()))
                    .expect("vec of float to 1d list");
                list.into_any()
            }

            InnerValue::VecBool(v, _) => {
                let list = PyList::new(py, v.into_iter().map(|x| x.into_pyobject(py).unwrap()))
                    .expect("vec of bool to 1d list");
                list.into_any()
            }

            InnerValue::VecText(v, _) => {
                let list = PyList::new(py, v.into_iter().map(|x| x.into_pyobject(py).unwrap()))
                    .expect("vec of str to 1d list");
                list.into_any()
            }

            InnerValue::MatrixInteger(m, _) => {
                let rows = PyList::new(
                    py,
                    m.into_iter().map(|row| {
                        PyList::new(py, row.into_iter().map(|x| x.into_pyobject(py).unwrap()))
                            .unwrap()
                    }),
                )
                .expect("2d int");
                rows.into_any()
            }

            InnerValue::MatrixFloat(m, _) => {
                let rows = PyList::new(
                    py,
                    m.into_iter().map(|row| {
                        PyList::new(py, row.into_iter().map(|x| x.into_pyobject(py).unwrap()))
                            .unwrap()
                    }),
                )
                .expect("2d float");
                rows.into_any()
            }

            InnerValue::MatrixBool(m, _) => {
                let rows = PyList::new(
                    py,
                    m.into_iter().map(|row| {
                        PyList::new(py, row.into_iter().map(|x| x.into_pyobject(py).unwrap()))
                            .unwrap()
                    }),
                )
                .expect("2d bool");
                rows.into_any()
            }

            InnerValue::MatrixText(m, _) => {
                let rows = PyList::new(
                    py,
                    m.into_iter().map(|row| {
                        PyList::new(py, row.into_iter().map(|x| x.into_pyobject(py).unwrap()))
                            .unwrap()
                    }),
                )
                .expect("2d str");
                rows.into_any()
            }
            InnerValue::Unsupported => py.None().into_bound(py),
        };
        Ok(obj)
    }
}

#[pyclass(str)]
#[pyo3(name = "Frame")]
struct PyFrame(Frame);

impl std::fmt::Display for PyFrame {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut buf = Vec::new();
        rs_write_frame(&mut buf, &self.0).map_err(|_| std::fmt::Error)?;
        let s = std::str::from_utf8(&buf).map_err(|_| std::fmt::Error)?;
        f.write_str(s)
    }
}

#[pymethods]
impl PyFrame {
    #[getter]
    fn natoms(self_: PyRef<'_, Self>) -> PyResult<u32> {
        Ok(self_.0.natoms())
    }

    #[getter]
    fn arrs(self_: PyRef<'_, Self>, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let dict = PyDict::new(py);
        let arrs = self_.0.arrs();
        for (k, v) in arrs {
            let v = Value(v.clone());
            dict.set_item(k, v.into_pyobject(py)?)?;
        }

        Ok(dict.into())
    }

    #[getter]
    fn info(self_: PyRef<'_, Self>, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let dict = PyDict::new(py);
        let arrs = self_.0.info();
        for (k, v) in arrs {
            let v = Value(v.clone());
            dict.set_item(k, v.into_pyobject(py)?)?;
        }

        Ok(dict.into())
    }

    fn __repr__(slf: &Bound<'_, Self>) -> PyResult<String> {
        let class_name: Bound<'_, PyString> = slf.get_type().qualname()?;
        Ok(format!("----- {} ----- : \n{}", class_name, slf))
    }
}

struct PyBinaryIO_R {
    // a TextIO
    obj: Py<PyAny>,
}

impl PyBinaryIO_R {
    fn new(stream: Py<PyAny>) -> Self {
        PyBinaryIO_R { obj: stream }
    }
}

impl std::io::Read for PyBinaryIO_R {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        Python::attach(|py| -> PyResult<usize> {
            let bytes = self
                .obj
                .call_method1(py, "read", (buf.len(),))?
                .extract::<&[u8]>(py)?
                .to_vec();

            let n = bytes.len();
            buf[..n].copy_from_slice(&bytes);
            Ok(n)
        })
        .map_err(|e| std::io::Error::other(e.to_string()))
    }
}

#[pyfunction]
#[pyo3(name = "read_frame")]
#[pyo3(text_signature = "(stream)")]
fn py_read_frame(_py: Python, stream: Py<PyAny>) -> PyResult<PyFrame> {
    let rd = PyBinaryIO_R::new(stream);
    let mut rd = BufReader::new(rd);
    let frame = rs_read_frame(&mut rd)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok(PyFrame(frame))
}

struct PyBinaryIO_W {
    // a TextIO as writer
    obj: Py<PyAny>,
}

impl PyBinaryIO_W {
    fn new(stream: Py<PyAny>) -> Self {
        PyBinaryIO_W { obj: stream }
    }
}

impl std::io::Write for PyBinaryIO_W {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        Python::attach(|py| -> PyResult<usize> {
            let n = self
                .obj
                .call_method1(py, "write", (buf,))?
                .extract::<usize>(py)?;
            Ok(n)
        })
        .map_err(|e| std::io::Error::other(e.to_string()))
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Python::attach(|py| -> PyResult<()> {
            self.obj.call_method0(py, "flush")?;
            Ok(())
        })
        .map_err(|e| std::io::Error::other(e.to_string()))
    }
}

#[pyfunction]
#[pyo3(name = "write_frame")]
#[pyo3(text_signature = "(stream, frame)")]
fn py_write_frame(_py: Python, stream: Py<PyAny>, frame: Bound<'_, PyFrame>) -> PyResult<()> {
    let mut w = PyBinaryIO_W::new(stream);
    rs_write_frame(&mut w, &frame.borrow().0)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok(())
}

#[pyclass]
struct PyFrameIteratorBinaryIO {
    inner: FrameReaderOwned<BufReader<PyBinaryIO_R>>,
}

#[pymethods]
impl PyFrameIteratorBinaryIO {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<'_, Self>) -> PyResult<Option<PyFrame>> {
        match slf.inner.next() {
            None => Ok(None),
            Some(Ok(frame)) => Ok(Some(PyFrame(frame))),
            Some(Err(err)) => Err(pyo3::exceptions::PyIOError::new_err(err.to_string())),
        }
    }
}

#[pyclass]
struct PyFrameIteratorFileReader {
    inner: FrameReaderOwned<BufReader<std::fs::File>>,
}

#[pymethods]
impl PyFrameIteratorFileReader {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<'_, Self>) -> PyResult<Option<PyFrame>> {
        match slf.inner.next() {
            None => Ok(None),
            Some(Ok(frame)) => Ok(Some(PyFrame(frame))),
            Some(Err(err)) => Err(pyo3::exceptions::PyIOError::new_err(err.to_string())),
        }
    }
}

#[pyfunction]
#[pyo3(name = "read_frames")]
#[pyo3(text_signature = "(stream)")]
fn py_read_frames(_py: Python, stream: Py<PyAny>) -> PyResult<PyFrameIteratorBinaryIO> {
    let rd = PyBinaryIO_R::new(stream);
    let rd = BufReader::new(rd);
    let frame_iter = PyFrameIteratorBinaryIO {
        inner: FrameReaderOwned::new(rd),
    };
    Ok(frame_iter)
}

#[pyfunction]
#[pyo3(name = "write_frames")]
#[pyo3(text_signature = "(stream, frames)")]
fn py_write_frames(py: Python, stream: Py<PyAny>, frames: Py<PyAny>) -> PyResult<usize> {
    let mut w = PyBinaryIO_W::new(stream);
    // TODO: can be combined because the inner impl is exactly the same, but I won't bother now.
    if let Ok(frames) = frames.extract::<Bound<PyFrameIteratorBinaryIO>>(py) {
        let mut frames = frames.borrow_mut();
        let mut count = 0;
        for frame in frames.inner.by_ref() {
            let frame =
                frame.map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            rs_write_frame(&mut w, &frame)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            count += 1;
        }
        Ok(count)
    } else if let Ok(frames) = frames.extract::<Bound<PyFrameIteratorFileReader>>(py) {
        let mut frames = frames.borrow_mut();
        let mut count = 0;
        for frame in frames.inner.by_ref() {
            let frame =
                frame.map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            rs_write_frame(&mut w, &frame)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            count += 1;
        }
        Ok(count)
    } else {
        Err(pyo3::exceptions::PyTypeError::new_err(
            "invalid frame iterator",
        ))
    }
}

#[pyfunction]
#[pyo3(name = "read_frame_from_file")]
#[pyo3(text_signature = "(inp)")]
fn py_read_frame_from_file(_py: Python, inp: PathBuf) -> PyResult<PyFrame> {
    let file = std::fs::OpenOptions::new()
        .read(true)
        .open(&inp)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    let mut rd = std::io::BufReader::new(file);
    let frame = rs_read_frame(&mut rd)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok(PyFrame(frame))
}

#[pyfunction]
#[pyo3(name = "read_frames_from_file")]
#[pyo3(text_signature = "(inp)")]
fn py_read_frames_from_file(_py: Python, inp: PathBuf) -> PyResult<PyFrameIteratorFileReader> {
    let file = std::fs::OpenOptions::new()
        .read(true)
        .open(&inp)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    let rd = std::io::BufReader::new(file);
    let frame_iter = PyFrameIteratorFileReader {
        inner: FrameReaderOwned::new(rd),
    };
    Ok(frame_iter)
}

#[pymodule]
#[pyo3(name = "extxyz")]
fn pyextxyz(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyFrame>()?;
    m.add_function(wrap_pyfunction!(py_read_frame, m)?)?;
    m.add_function(wrap_pyfunction!(py_read_frames, m)?)?;
    m.add_function(wrap_pyfunction!(py_write_frame, m)?)?;
    m.add_function(wrap_pyfunction!(py_write_frames, m)?)?;
    m.add_function(wrap_pyfunction!(py_read_frame_from_file, m)?)?;
    m.add_function(wrap_pyfunction!(py_read_frames_from_file, m)?)?;
    Ok(())
}
