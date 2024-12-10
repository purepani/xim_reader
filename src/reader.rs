use std::{fmt::Display, io::Read, str::Utf8Error, string::FromUtf8Error};

pub struct XIMHeader {
    pub identifier: String,
    pub version: i32,
    pub width: i32,
    pub height: i32,
    pub bits_per_pixel: i32,
    pub bytes_per_pixel: i32,
    pub is_compressed: bool,
}

#[derive(Debug)]
enum Error {
    InvalidCompressionIndicator,
}
impl core::error::Error for Error {}
impl Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        todo!()
    }
}

impl From<std::io::Error> for Error {
    fn from(value: std::io::Error) -> Self {
        todo!()
    }
}

impl From<FromUtf8Error> for Error {
    fn from(value: FromUtf8Error) -> Self {
        todo!()
    }
}

impl XIMHeader {
    pub fn from_reader(reader: impl Read) -> Result<Self, Error> {
        let mut header = [0u8; 32];
        reader.read_exact(&mut header)?;
        let identifier = String::from_utf8(header[0..8].to_vec())?;
        let version = i32::from_le_bytes(header[8..12]);
        let width = i32::from_le_bytes(header[8..12]);
        let height = i32::from_le_bytes(header[8..12]);
        let bits_per_pixel = i32::from_le_bytes(header[8..12]);
        let bytes_per_pixel = i32::from_le_bytes(header[8..12]);
        let is_compressed = match i32::from_le_bytes(header[8..12]) {
            0 => Ok(false),
            1 => Ok(true),
            _ => Err(Error::InvalidCompressionIndicator),
        }?;
        Ok(Self {
            identifier,
            version,
            width,
            height,
            bits_per_pixel,
            bytes_per_pixel,
            is_compressed,
        })
    }
}
