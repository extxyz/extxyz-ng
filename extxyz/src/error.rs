/// Convenient result type used throughout the crate.
///
/// This is a type alias for [`std::result::Result`] where the error type is
/// [`ExtxyzError`]. It simplifies function signatures when working with
/// extended XYZ parsing and writing.
/// ```
pub type Result<T> = std::result::Result<T, ExtxyzError>;

/// Errors that can occur while reading or writing extended XYZ data.
///
/// This enum represents the different failure modes encountered during
/// parsing, validation, or I/O operations.
///
/// # Variants
/// - `Io`: Wraps an underlying [`std::io::Error`] that occurred during
///   reading or writing.
/// - `InvalidValue`: Indicates that a value could not be parsed or does
///   not conform to the expected format. Contains a static error message
///   describing the issue.
///
/// # Notes
/// - `InvalidValue` is typically used for semantic or format-related errors,
///   while `Io` represents lower-level system I/O failures.
/// - Additional variants may be added in the future as the parser evolves.
#[derive(Debug)]
pub enum ExtxyzError {
    /// An error originating from I/O operations.
    Io(std::io::Error),

    /// A value is invalid or cannot be parsed.
    InvalidValue(&'static str),
}

impl std::fmt::Display for ExtxyzError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExtxyzError::Io(error) => write!(f, "{error}"),
            ExtxyzError::InvalidValue(msg) => write!(f, "{msg}"),
        }
    }
}

impl std::error::Error for ExtxyzError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            ExtxyzError::Io(error) => Some(error),
            ExtxyzError::InvalidValue(_) => None,
        }
    }
}

impl From<std::io::Error> for ExtxyzError {
    fn from(value: std::io::Error) -> Self {
        ExtxyzError::Io(value)
    }
}
