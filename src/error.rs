//! The crate's error type.

/// Errors that can occur during compilation.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
#[must_use]
pub enum Error {
    /// The function uses a feature this backend does not (yet) support.
    Unsupported(String),
    /// The environment could not resolve a reference.
    Env(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Unsupported(s) => write!(f, "unsupported: {s}"),
            Error::Env(s) => write!(f, "environment error: {s}"),
        }
    }
}

impl std::error::Error for Error {}

/// A `Result` whose error type is this crate's [`Error`].
pub type Result<T> = std::result::Result<T, Error>;

pub(crate) fn unsupported<T>(msg: impl Into<String>) -> Result<T> {
    Err(Error::Unsupported(msg.into()))
}
