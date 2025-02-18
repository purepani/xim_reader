use numpy::PyArray2;
use numpy::PyArrayMethods;
use pyo3::{prelude::Bound, pyclass, pymethods};
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use std::{
    convert::TryInto,
    fmt::Display,
    fs::File,
    io::{BufReader, Read},
    path::PathBuf,
    string::FromUtf8Error,
};

use ndarray::{ArrayViewMut2, ShapeError};

#[gen_stub_pyclass]
#[pyclass]
pub struct XIMImage {
    header: XIMHeader,
    pixel_data: PixelData,
    histogram: XIMHistogram,
    properties: XIMProperties,
}

#[gen_stub_pymethods]
#[pymethods]
impl XIMImage {
    #[new]
    pub fn new(image_path: PathBuf) -> Self {
        let file = File::open(image_path).unwrap();
        let mut reader = BufReader::new(file);
        let header = XIMHeader::from_reader(&mut reader).unwrap();
        let pixel_data = if header.is_compressed {
            PixelData::from_compressed(&mut reader, header.clone()).unwrap()
        } else {
            PixelData::from_uncompressed(&mut reader, header.clone()).unwrap()
        };
        let histogram = XIMHistogram::from_reader(&mut reader).unwrap();
        let properties = XIMProperties::from_reader(&mut reader).unwrap();
        Self {
            header,
            pixel_data,
            histogram,
            properties,
        }
    }

    pub fn numpy<'py>(this: Bound<'py, Self>) -> Bound<'py, PyArray2<i16>> {
        let array = &this.borrow().pixel_data.0;
        unsafe {
            let pyarray = PyArray2::borrow_from_array_bound(array, this.into_any());
            pyarray.readwrite().make_nonwriteable();
            pyarray
        }
    }
}

#[derive(Debug, Clone)]
pub struct XIMHeader {
    pub identifier: String,
    pub version: i32,
    pub width: i32,
    pub height: i32,
    pub bits_per_pixel: i32,
    pub bytes_per_pixel: i32,
    pub is_compressed: bool,
}

#[derive(Debug, Clone)]
pub struct PixelData(ndarray::Array2<i16>);

#[derive(Debug, Clone)]
pub struct XIMHistogram {
    pub number_of_bins: i32,
    pub histogram: Vec<i32>,
}

#[derive(Debug, Clone)]
pub enum PropertyType {
    Integer,
    Double,
    String,
    DoubleArray,
    IntegerArray,
}

#[derive(Debug, Clone)]
pub enum PropertyValue {
    Integer(i32),
    Double(f64),
    String(String),
    DoubleArray(Vec<f64>),
    IntegerArray(Vec<i32>),
}

#[derive(Debug, Clone)]
pub struct Property {
    pub property_name_length: i32,
    pub property_name: String,
    pub property_type: PropertyType,
    pub property_value: PropertyValue,
}

#[derive(Debug, Clone)]
pub struct XIMProperties {
    pub num_properties: i32,
    pub properties: Vec<Property>,
}

impl XIMHeader {
    pub fn from_reader<R: Read>(reader: &mut R) -> Result<Self, Error> {
        let mut identifier = [0u8; 8];
        let mut version = [0u8; 4];
        let mut width = [0u8; 4];
        let mut height = [0u8; 4];
        let mut bits_per_pixel = [0u8; 4];
        let mut bytes_per_pixel = [0u8; 4];
        let mut compression = [0u8; 4];

        reader.read_exact(&mut identifier)?;
        reader.read_exact(&mut version)?;
        reader.read_exact(&mut width)?;
        reader.read_exact(&mut height)?;
        reader.read_exact(&mut bits_per_pixel)?;
        reader.read_exact(&mut bytes_per_pixel)?;
        reader.read_exact(&mut compression)?;

        let identifier = String::from_utf8(identifier.to_vec())?;
        let version = i32::from_le_bytes(version);
        let width = i32::from_le_bytes(width);
        let height = i32::from_le_bytes(height);
        let bits_per_pixel = i32::from_le_bytes(bits_per_pixel);
        let bytes_per_pixel = i32::from_le_bytes(bytes_per_pixel);
        let is_compressed = match i32::from_le_bytes(compression) {
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

impl PixelData {
    pub fn from_uncompressed(mut reader: impl Read, header: XIMHeader) -> Result<Self, Error> {
        if header.width < 0 {
            return Err(Error::InvalidWidth);
        }
        if header.height < 0 {
            return Err(Error::InvalidHeight);
        }
        let width = header.width as usize;
        let height = header.height as usize;

        let mut pixel_buffer_size = [0u8; 4];
        reader.read_exact(&mut pixel_buffer_size)?;
        let pixel_buffer_size = i32::from_le_bytes(pixel_buffer_size);

        if pixel_buffer_size < 0 {
            return Err(Error::InvalidPixelBufferSize);
        }
        let mut data: Vec<i16> = Vec::with_capacity(width * height);
        for _i in 0..(width * height) {
            let mut buf = [0u8; 2];
            reader.read_exact(&mut buf)?;
            data.push(i16::from_le_bytes(buf));
        }

        let array = ndarray::Array2::from_shape_vec((width, height), data)?;
        Ok(Self(array))
    }
    pub fn parse_lookup(lookup_table: Vec<u8>) -> Vec<u8> {
        let num_bytes_table = lookup_table
            .into_iter()
            .flat_map(|vals| {
                vec![
                    (vals & 0b00000011),
                    (vals & 0b00001100) >> 2,
                    (vals & 0b00110000) >> 4,
                    (vals & 0b11000000) >> 6,
                ]
            })
            .map(|val| 1u8 << val)
            .collect::<Vec<u8>>();
        num_bytes_table
    }

    pub fn from_compressed(reader: &mut impl Read, header: XIMHeader) -> Result<Self, Error> {
        if header.width < 0 {
            return Err(Error::InvalidWidth);
        }
        if header.height < 0 {
            return Err(Error::InvalidHeight);
        }
        let width = header.width as usize;
        let height = header.height as usize;

        let mut lookup_table_size = [0u8; 4];
        reader.read_exact(&mut lookup_table_size).unwrap();
        let lookup_table_size = i32::from_le_bytes(lookup_table_size);

        let mut lookup_table: Vec<u8> = vec![0u8; lookup_table_size.try_into().unwrap()];
        reader.read_exact(&mut lookup_table).unwrap();

        let mut compressed_pixel_buffer_size = [0u8; 4];
        reader
            .read_exact(&mut compressed_pixel_buffer_size)
            .unwrap();
        let compressed_pixel_buffer_size = i32::from_le_bytes(compressed_pixel_buffer_size);

        if compressed_pixel_buffer_size < 0 {
            return Err(Error::InvalidPixelBufferSize);
        }
        let compressed_pixel_buffer_size = compressed_pixel_buffer_size.unsigned_abs();

        let num_bytes_table = Self::parse_lookup(lookup_table);
        let full_len = num_bytes_table.len();
        let num_bytes_table = num_bytes_table
            .into_iter()
            .take(full_len - (width * (height - 1)) % 4 - 1);

        let compressed_pixel_buffer = {
            let mut buf = vec![0; compressed_pixel_buffer_size.try_into().unwrap()];
            let _ = reader.read_exact(&mut buf);
            buf.into_iter()
        }
        .collect::<Vec<_>>();

        let (uncompressed_buffer, compressed_diffs) =
            compressed_pixel_buffer.split_at((width + 1) * 4);
        let mut compressed_diffs = compressed_diffs.into_iter();

        let initial_uncompressed = uncompressed_buffer
            .into_iter()
            .map(|val| *val)
            .collect::<Vec<u8>>()
            .chunks_exact(4)
            .map(|val| {
                let mut buf = [0u8; 4];
                buf.copy_from_slice(val);
                i32::from_le_bytes(buf) as i16
            })
            .collect::<Vec<_>>();

        let differences = num_bytes_table
            .map(|num_bytes| {
                let mut val = compressed_diffs
                    .by_ref()
                    .take(num_bytes.try_into().unwrap())
                    .map(|val| *val);
                match num_bytes {
                    1 => {
                        let mut buf = [0; 1];
                        buf.fill_with(|| val.next().unwrap_or(0));
                        i16::from(i8::from_le_bytes(buf))
                    }
                    2 => {
                        let mut buf = [0; 2];
                        buf.fill_with(|| val.next().unwrap_or(0));
                        i16::from_le_bytes(buf)
                    }
                    4 => {
                        let mut buf = [0; 4];
                        buf.fill_with(|| val.next().unwrap_or(0));
                        i32::from_le_bytes(buf) as i16
                    }
                    _ => todo!(),
                }
            })
            .collect::<Vec<_>>();

        let uncompressed_data = [initial_uncompressed, differences].concat();
        let mut array =
            ndarray::Array2::from_shape_vec((height, width), uncompressed_data).unwrap();
        Self::decompress_diffs(array.view_mut());

        let mut buf = [0u8; 4];
        let _ = reader.read_exact(&mut buf);

        Ok(Self(array))
    }

    pub fn decompress_diffs(mut compressed_arr: ArrayViewMut2<i16>) -> ArrayViewMut2<i16> {
        let width = compressed_arr.ncols();

        let arr = compressed_arr.as_slice_mut().unwrap();

        let first_index = width + 1;
        for i in first_index..arr.len() {
            let left = *arr.get(i - 1).unwrap();
            let above = *arr.get(i - width).unwrap();
            let upper_left = *arr.get(i - width - 1).unwrap();
            let diff = arr.get_mut(i).unwrap();
            *diff = diff.wrapping_add(left);
            *diff = diff.wrapping_add(above);
            *diff = diff.wrapping_sub(upper_left);
        }
        compressed_arr
    }
}

#[derive(Debug)]
pub enum Error {
    InvalidCompressionIndicator,
    InvalidWidth,
    InvalidHeight,
    InvalidPixelBufferSize,
    InvalidOther(String),
}
impl core::error::Error for Error {}
impl Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::InvalidCompressionIndicator => todo!(),
            Error::InvalidWidth => todo!(),
            Error::InvalidHeight => todo!(),
            Error::InvalidPixelBufferSize => todo!(),
            Error::InvalidOther(val) => write!(f, "Failed: {}", val),
        }
    }
}

impl From<ShapeError> for Error {
    fn from(value: ShapeError) -> Self {
        todo!()
    }
}

impl From<std::io::Error> for Error {
    fn from(value: std::io::Error) -> Self {
        Self::InvalidOther(value.to_string())
    }
}

impl From<FromUtf8Error> for Error {
    fn from(value: FromUtf8Error) -> Self {
        todo!()
    }
}

impl XIMHistogram {
    pub fn from_reader<R: Read>(reader: &mut R) -> Result<Self, Error> {
        let mut number_of_bins = [0u8; 4];

        reader.read_exact(&mut number_of_bins)?;
        let number_of_bins = i32::from_le_bytes(number_of_bins);

        let mut histogram = vec![0u8; (number_of_bins * 4).try_into().unwrap()];
        reader.read_exact(&mut histogram);

        let histogram = histogram
            .chunks_exact(4)
            .into_iter()
            .map(|val| {
                let mut buf = [0u8; 4];
                buf.copy_from_slice(val);
                i32::from_le_bytes(buf)
            })
            .collect::<Vec<_>>();
        Ok(Self {
            number_of_bins,
            histogram,
        })
    }
}

impl Property {
    pub fn from_reader<R: Read>(reader: &mut R) -> Result<Self, Error> {
        let mut property_name_length = [0u8; 4];
        reader.read_exact(&mut property_name_length)?;
        let property_name_length = i32::from_le_bytes(property_name_length);

        let mut property_name = vec![0u8; property_name_length.try_into().unwrap()];
        reader.read_exact(&mut property_name)?;
        let property_name = String::from_utf8(property_name)?;

        let mut property_type = [0u8; 4];
        reader.read_exact(&mut property_type)?;
        let property_type = i32::from_le_bytes(property_type);
        let property_type = match property_type {
            0 => PropertyType::Integer,
            1 => PropertyType::Double,
            2 => PropertyType::String,
            4 => PropertyType::DoubleArray,
            5 => PropertyType::IntegerArray,
            _ => todo!(),
        };

        let property_value = match property_type {
            PropertyType::Integer => {
                let mut value = [0u8; 4];
                reader.read_exact(&mut value)?;
                let value = i32::from_le_bytes(value);
                PropertyValue::Integer(value)
            }
            PropertyType::Double => {
                let mut value = [0u8; 8];
                reader.read_exact(&mut value)?;
                let value = f64::from_le_bytes(value);
                PropertyValue::Double(value)
            }
            PropertyType::String => {
                let mut value_len = [0u8; 4];
                reader.read_exact(&mut value_len)?;
                let value_len = i32::from_le_bytes(value_len);

                let mut value = vec![0u8; value_len.try_into().unwrap()];
                reader.read_exact(&mut value)?;
                let value = String::from_utf8(value).unwrap();
                PropertyValue::String(value)
            }
            PropertyType::DoubleArray => {
                let mut value_len = [0u8; 4];
                reader.read_exact(&mut value_len)?;
                let value_len = i32::from_le_bytes(value_len);

                let mut value = vec![0u8; value_len.try_into().unwrap()];
                reader.read_exact(&mut value)?;
                let value = value
                    .chunks_exact(8)
                    .map(|val| {
                        let mut buf = [0u8; 8];
                        buf.copy_from_slice(val);
                        f64::from_le_bytes(buf)
                    })
                    .collect::<Vec<_>>();
                PropertyValue::DoubleArray(value)
            }
            PropertyType::IntegerArray => {
                let mut value_len = [0u8; 4];
                reader.read_exact(&mut value_len)?;
                let value_len = i32::from_le_bytes(value_len);

                let mut value = vec![0u8; value_len.try_into().unwrap()];
                reader.read_exact(&mut value)?;
                let value = value
                    .chunks_exact(4)
                    .map(|val| {
                        let mut buf = [0u8; 4];
                        buf.copy_from_slice(val);
                        i32::from_le_bytes(buf)
                    })
                    .collect::<Vec<_>>();
                PropertyValue::IntegerArray(value)
            }
        };

        Ok(Self {
            property_name_length,
            property_name,
            property_type,
            property_value,
        })
    }
}

impl XIMProperties {
    pub fn from_reader<R: Read>(reader: &mut R) -> Result<Self, Error> {
        let mut num_properties = [0u8; 4];
        reader.read_exact(&mut num_properties).unwrap();
        let num_properties = i32::from_le_bytes(num_properties);

        let mut properties = Vec::with_capacity(num_properties.try_into().unwrap());

        for i in 0..num_properties {
            let property = Property::from_reader(reader).unwrap();
            properties.push(property)
        }
        Ok(Self {
            num_properties,
            properties,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decompression() {
        let input: [i16; 8] = [4, 3, 10, 1, 10, 30, 20, 40];
        let mut input_array = ndarray::Array2::from_shape_vec((4, 2), input.to_vec()).unwrap();
        let calculated_output = PixelData::decompress_diffs(input_array.view_mut());
        let output =
            ndarray::Array2::from_shape_vec((4, 2), vec![4, 3, 10, 10, 27, 57, 94, 164]).unwrap();
        assert_eq!(calculated_output, output);
    }
    #[test]
    fn test_parse_lookup() {
        let test: Vec<u8> = vec![1, 10, 30, 20, 40];
        println!("{:#010b}", test.get(1).unwrap());
        let test = test
            .into_iter()
            .map(|val| val.to_le_bytes().to_vec())
            .collect::<Vec<_>>()
            .concat();
        let calculated_output = PixelData::parse_lookup(test);
        let output: Vec<u8> = vec![2, 1, 1, 1, 4, 4, 1, 1, 4, 8, 2, 1, 1, 2, 2, 1, 1, 4, 4, 1];
        assert_eq!(output, calculated_output);
    }
}
