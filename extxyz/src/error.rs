pub type Result<T> = std::result::Result<T, ExtxyzError>;

#[derive(Debug)]
pub enum ExtxyzError {
    Io(std::io::Error),
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
