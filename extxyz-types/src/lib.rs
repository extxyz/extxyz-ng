#![allow(clippy::match_bool)]

use std::{borrow::Cow, collections::HashMap, ops::Deref};

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
/// use extxyz_types::Integer;
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
/// use extxyz_types::FloatNum;
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
/// use extxyz_types::Boolean;
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
/// use extxyz_types::Text;
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

// Safe hardler for `DictEntry`
#[derive(Debug)]
pub struct DictHandler(pub Vec<(String, Value)>);

impl DictHandler {
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

/// A raw frame parsed from an `extxyz` file.
///
/// This struct represents the data for a single frame of an `extxyz` file,
/// including the number of atoms, metadata, and per-atom arrays.  
///
/// You can iterate over the per-atom arrays directly:
/// ```ignore
/// for (name, value) in frame.arrs() {
///     println!("{name}: {value:?}");
/// }
/// ```
///
/// Or convert the metadata info into a `HashMap` for easy lookup:
/// ```ignore
/// let info_map = frame.info();
/// if let Some(temperature) = info_map.get("temperature") {
///     println!("Temperature: {:?}", temperature);
/// }
/// ```
pub struct Frame {
    pub natoms: u32,
    pub info: DictHandler,
    pub arrs: DictHandler,
}

impl Frame {
    /// Returns the number of atoms in the frame.
    #[must_use]
    pub fn natoms(&self) -> u32 {
        self.natoms
    }

    /// Returns the frame metadata (`info`) as a `HashMap` for easy lookup.
    ///
    /// Keys are `&str` slices pointing to the original `String`s inside
    /// `DictHandler`, and values are references to `Value`.
    ///
    /// # Example
    /// ```ignore
    /// let arrs_map = frame.arrs();
    /// if let Some(pos) = arrs_map.get("pos") {
    ///     println!("Positions: {:?}", pos);
    /// }
    /// ```
    #[must_use]
    pub fn arrs(&self) -> HashMap<&str, &Value> {
        let arrs = self.arrs.iter().map(|(k, v)| (k.as_str(), v));
        arrs.collect::<HashMap<_, _>>()
    }

    /// Returns the frame metadata (`info`) as a `HashMap` for easy lookup.
    ///
    /// Keys are `&str` slices pointing to the original `String`s inside
    /// `DictHandler`, and values are references to `Value`.
    ///
    /// # Example
    /// ```ignore
    /// let info_map = frame.info();
    /// if let Some(temperature) = info_map.get("temperature") {
    ///     println!("Temperature: {:?}", temperature);
    /// }
    /// ```
    #[must_use]
    pub fn info(&self) -> HashMap<&str, &Value> {
        let info = self.info.iter().map(|(k, v)| (k.as_str(), v));
        info.collect::<HashMap<_, _>>()
    }
}
