use byteorder::ByteOrder;
use core::num;
use numpy::PyArray2;
use numpy::PyArrayMethods;
use pyo3::{prelude::Bound, pyclass, pymethods, IntoPyObject};
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pyclass_enum, gen_stub_pymethods};
use pyo3_stub_gen::impl_stub_type;
use std::collections::HashMap;
use std::fmt::Debug;
use std::{
    convert::TryInto,
    fmt::Display,
    fs::File,
    io::{BufReader, Read},
    path::PathBuf,
    string::FromUtf8Error,
};

use byteorder::{ReadBytesExt, LE};
use ndarray::{ArrayViewMut2, ShapeError};

trait ReadFromReader<I> {
    type Error;
    fn read_type_into<B: ByteOrder>(&mut self, dst: &mut [I]) -> Result<(), Self::Error>;
    fn read_type<B: ByteOrder>(&mut self) -> Result<I, Self::Error>;
}

impl<R: ReadBytesExt> ReadFromReader<i8> for R {
    type Error = std::io::Error;
    fn read_type_into<B: ByteOrder>(&mut self, dst: &mut [i8]) -> Result<(), std::io::Error> {
        self.read_i8_into(dst)
    }

    fn read_type<B: ByteOrder>(&mut self) -> Result<i8, std::io::Error> {
        self.read_i8()
    }
}

impl<R: ReadBytesExt> ReadFromReader<i16> for R {
    type Error = std::io::Error;
    fn read_type_into<B: ByteOrder>(&mut self, dst: &mut [i16]) -> Result<(), std::io::Error> {
        self.read_i16_into::<B>(dst)
    }

    fn read_type<B: ByteOrder>(&mut self) -> Result<i16, std::io::Error> {
        self.read_i16::<B>()
    }
}

impl<R: ReadBytesExt> ReadFromReader<i32> for R {
    type Error = std::io::Error;
    fn read_type_into<B: ByteOrder>(&mut self, dst: &mut [i32]) -> Result<(), std::io::Error> {
        self.read_i32_into::<B>(dst)
    }

    fn read_type<B: ByteOrder>(&mut self) -> Result<i32, std::io::Error> {
        self.read_i32::<B>()
    }
}

impl<R: ReadBytesExt> ReadFromReader<i64> for R {
    type Error = std::io::Error;
    fn read_type_into<B: ByteOrder>(&mut self, dst: &mut [i64]) -> Result<(), std::io::Error> {
        self.read_i64_into::<B>(dst)
    }

    fn read_type<B: ByteOrder>(&mut self) -> Result<i64, std::io::Error> {
        self.read_i64::<B>()
    }
}

#[derive(IntoPyObject)]
pub enum XIMArray<'py> {
    #[pyo3(transparent)]
    Int8(Bound<'py, PyArray2<i8>>),
    #[pyo3(transparent)]
    Int16(Bound<'py, PyArray2<i16>>),
    #[pyo3(transparent)]
    Int32(Bound<'py, PyArray2<i32>>),
    #[pyo3(transparent)]
    Int64(Bound<'py, PyArray2<i64>>),
}

impl_stub_type!(XIMArray<'_> = PyArray2<i8> | PyArray2<i16> |PyArray2<i32> |PyArray2<i64>);

#[gen_stub_pyclass]
#[pyclass]
pub struct XIMImage {
    #[pyo3(get)]
    pub header: XIMHeader,
    pixel_data: PixelDataSupported,
    pub histogram: XIMHistogram,
    pub properties: XIMProperties,
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
            PixelDataSupported::from_compressed(&mut reader, &header).unwrap()
        } else {
            PixelDataSupported::from_uncompressed(&mut reader, &header).unwrap()
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

    #[getter]
    pub fn numpy<'py>(this: Bound<'py, Self>) -> XIMArray<'py> {
        match &this.borrow().pixel_data {
            PixelDataSupported::Int8(pixel_data) => {
                let array = &pixel_data.0;
                unsafe {
                    let pyarray = PyArray2::borrow_from_array(array, this.into_any());
                    pyarray.readwrite().make_nonwriteable();
                    XIMArray::Int8(pyarray)
                }
            }
            PixelDataSupported::Int16(pixel_data) => {
                let array = &pixel_data.0;
                unsafe {
                    let pyarray = PyArray2::borrow_from_array(array, this.into_any());
                    pyarray.readwrite().make_nonwriteable();
                    XIMArray::Int16(pyarray)
                }
            }
            PixelDataSupported::Int32(pixel_data) => {
                let array = &pixel_data.0;
                unsafe {
                    let pyarray = PyArray2::borrow_from_array(array, this.into_any());
                    pyarray.readwrite().make_nonwriteable();
                    XIMArray::Int32(pyarray)
                }
            }
            PixelDataSupported::Int64(pixel_data) => {
                let array = &pixel_data.0;
                unsafe {
                    let pyarray = PyArray2::borrow_from_array(array, this.into_any());
                    pyarray.readwrite().make_nonwriteable();
                    XIMArray::Int64(pyarray)
                }
            }
        }
    }

    #[getter]
    pub fn histogram(&self) -> Vec<i32> {
        self.histogram.histogram.clone()
    }

    #[getter]
    pub fn properties(&self) -> HashMap<String, PropertyValue> {
        self.properties.properties.clone()
    }
}

#[derive(Debug, Clone)]
#[gen_stub_pyclass]
#[pyclass]
pub struct XIMHeader {
    #[pyo3(get)]
    pub identifier: String,
    #[pyo3(get)]
    pub version: i32,
    #[pyo3(get)]
    pub width: i32,
    #[pyo3(get)]
    pub height: i32,
    #[pyo3(get)]
    pub bits_per_pixel: i32,
    #[pyo3(get)]
    pub bytes_per_pixel: i32,
    #[pyo3(get)]
    pub is_compressed: bool,
}

#[derive(Debug, Clone)]
pub struct PixelData<I>(ndarray::Array2<I>);

#[derive(Debug, Clone)]
pub enum PixelDataSupported {
    Int8(PixelData<i8>),
    Int16(PixelData<i16>),
    Int32(PixelData<i32>),
    Int64(PixelData<i64>),
}

#[derive(Debug, Clone)]
#[gen_stub_pyclass]
#[pyclass]
pub struct XIMHistogram {
    #[pyo3(get)]
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

#[derive(Debug, Clone, IntoPyObject)]
pub enum PropertyValue {
    #[pyo3(transparent)]
    Integer(i32),
    #[pyo3(transparent)]
    Double(f64),
    #[pyo3(transparent)]
    String(String),
    #[pyo3(transparent)]
    DoubleArray(Vec<f64>),
    #[pyo3(transparent)]
    IntegerArray(Vec<i32>),
}

impl_stub_type!(PropertyValue = i32 | f64 | String | Vec<f64> | Vec<i32>);

#[derive(Debug, Clone)]
pub struct Property {
    property_name_length: i32,
    pub property_name: String,
    property_type: PropertyType,
    pub property_value: PropertyValue,
}

#[derive(Debug, Clone)]
#[gen_stub_pyclass]
#[pyclass]
pub struct XIMProperties {
    pub properties: HashMap<String, PropertyValue>,
}

impl XIMHeader {
    pub fn from_reader<R: Read>(reader: &mut R) -> Result<Self, Error> {
        let mut identifier = [0u8; 8];
        reader.read_exact(&mut identifier)?;
        let identifier = String::from_utf8(identifier.to_vec())?;

        let mut values = [0i32; 6];
        let _ = reader.read_i32_into::<LE>(&mut values)?;
        let [version, width, height, bits_per_pixel, bytes_per_pixel, compression] = values;

        let is_compressed = match compression {
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

    pub fn width(&self) -> Result<usize, Error> {
        usize::try_from(self.width).map_err(|_err| Error::InvalidWidth)
    }

    pub fn height(&self) -> Result<usize, Error> {
        usize::try_from(self.height).map_err(|_err| Error::InvalidHeight)
    }
}

struct LookupTableSequenceItem {
    pub key: usize,
    pub sequence_length: usize,
}

impl<I> PixelData<I> {
    pub fn new(array: ndarray::Array2<I>) -> Self {
        Self(array)
    }
}

impl PixelDataSupported {
    fn read_to_arr<I, R>(mut reader: R, width: usize, height: usize) -> Result<PixelData<I>, Error>
    where
        I: num_traits::ConstZero + Clone + Copy,
        R: Read + ReadFromReader<I>,
    {
        let mut data: Vec<I> = vec![I::ZERO; width * height];
        ReadFromReader::<I>::read_type_into::<LE>(&mut reader, &mut data);
        let array = ndarray::Array2::from_shape_vec((width, height), data)?;
        Ok(PixelData::new(array))
    }

    pub fn from_uncompressed(mut reader: impl Read, header: &XIMHeader) -> Result<Self, Error> {
        let num_bytes = header.bytes_per_pixel;

        let width = header.width()?;
        let height = header.height()?;

        let pixel_buffer_size = reader.read_i32::<LE>()?;
        let _pixel_buffer_size =
            usize::try_from(pixel_buffer_size).map_err(|err| Error::InvalidPixelBufferSize)?;
        match num_bytes {
            1 => Self::read_to_arr(&mut reader, width, height).map(Self::Int8),
            2 => Self::read_to_arr(&mut reader, width, height).map(Self::Int16),
            4 => Self::read_to_arr(&mut reader, width, height).map(Self::Int32),
            8 => Self::read_to_arr(&mut reader, width, height).map(Self::Int64),
            _ => todo!(),
        }
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

    fn decompress_array<I>(
        compressed_pixel_buffer: Vec<u8>,
        num_bytes_table: impl Iterator<Item = u8>,
        width: usize,
        height: usize,
    ) -> Result<PixelData<I>, Error>
    where
        I: num_traits::ConstZero
            + TryFrom<i32>
            + TryFrom<i8>
            + TryFrom<i16>
            + Clone
            + Copy
            + num_traits::WrappingAdd
            + num_traits::WrappingSub,
    {
        let (mut uncompressed_buffer, mut compressed_diffs) =
            compressed_pixel_buffer.split_at((width + 1) * 4);

        let initial_uncompressed = {
            let mut initial_uncompressed = vec![0i32; width + 1];
            uncompressed_buffer.read_i32_into::<LE>(&mut initial_uncompressed)?;
            initial_uncompressed
                .into_iter()
                .map(|val| {
                    I::try_from(val)
                        .map_err(|err| Error::InvalidPixels)
                        .unwrap()
                })
                .collect::<Vec<I>>()
        };

        let differences = num_bytes_table
            .map(|num_bytes| match num_bytes {
                1 => compressed_diffs
                    .read_i8()
                    .map_err(|err| Error::InvalidPixels)
                    .and_then(|x| I::try_from(x).map_err(|err| Error::InvalidPixels))
                    .unwrap(),
                2 => compressed_diffs
                    .read_i16::<LE>()
                    .map_err(|err| Error::InvalidPixels)
                    .and_then(|x| I::try_from(x).map_err(|err| Error::InvalidPixels))
                    .unwrap(),
                4 => compressed_diffs
                    .read_i32::<LE>()
                    .map_err(|err| Error::InvalidPixels)
                    .and_then(|x| I::try_from(x).map_err(|err| Error::InvalidPixels))
                    .unwrap(),
                _ => todo!(),
            })
            .collect::<Vec<I>>();

        let array = {
            let uncompressed_data = [initial_uncompressed, differences].concat();
            let mut array =
                ndarray::Array2::from_shape_vec((height, width), uncompressed_data).unwrap();
            Self::decompress_diffs(array.view_mut());
            array
        };
        Ok(PixelData::new(array))
    }

    pub fn from_compressed(reader: &mut impl Read, header: &XIMHeader) -> Result<Self, Error> {
        let width = header.width()?;
        let height = header.height()?;

        let lookup_table: Vec<u8> = {
            let lookup_table_size = reader.read_i32::<LE>()?;
            let mut buf = vec![0u8; lookup_table_size.try_into().unwrap()];
            reader.read_exact(&mut buf).unwrap();
            buf
        };
        let compressed_pixel_buffer_size = reader.read_i32::<LE>()?;
        let compressed_pixel_buffer_size = usize::try_from(compressed_pixel_buffer_size)
            .map_err(|_err| Error::InvalidPixelBufferSize)?;

        let num_bytes_table = {
            let num_bytes_table = Self::parse_lookup(lookup_table);
            let full_len = num_bytes_table.len();
            num_bytes_table
                .into_iter()
                .take(full_len - (width * (height - 1)) % 4 - 1)
        };

        let compressed_pixel_buffer = {
            let mut buf = vec![0; compressed_pixel_buffer_size];
            let _ = reader.read_exact(&mut buf);
            buf
        };

        let pixel_data = match header.bytes_per_pixel {
            1 => {
                let arr = Self::decompress_array(
                    compressed_pixel_buffer,
                    num_bytes_table,
                    width,
                    height,
                )?;
                PixelDataSupported::Int8(arr)
            }
            2 => {
                let arr = Self::decompress_array(
                    compressed_pixel_buffer,
                    num_bytes_table,
                    width,
                    height,
                )?;
                PixelDataSupported::Int16(arr)
            }
            4 => {
                let arr = Self::decompress_array(
                    compressed_pixel_buffer,
                    num_bytes_table,
                    width,
                    height,
                )?;
                PixelDataSupported::Int32(arr)
            }
            _ => todo!(),
        };

        let _uncompressed_buffer_size = reader.read_i32::<LE>()?;
        Ok(pixel_data)
    }

    pub fn decompress_diffs<I>(mut compressed_arr: ArrayViewMut2<I>) -> ArrayViewMut2<I>
    where
        I: num_traits::WrappingAdd + num_traits::WrappingSub + Copy,
    {
        let width = compressed_arr.ncols();

        let arr = compressed_arr.as_slice_mut().unwrap();

        let first_index = width + 1;
        for i in first_index..arr.len() {
            let left = *arr.get(i - 1).unwrap();
            let above = *arr.get(i - width).unwrap();
            let upper_left = *arr.get(i - width - 1).unwrap();
            let diff = arr.get_mut(i).unwrap();
            *diff = diff.wrapping_add(&left);
            *diff = diff.wrapping_add(&above);
            *diff = diff.wrapping_sub(&upper_left);
        }
        compressed_arr
    }
}

#[derive(Debug)]
pub enum Error {
    InvalidPixels,
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
            Error::InvalidPixels => todo!(),
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
        Ok(Self { histogram })
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
        let num_properties = reader
            .read_i32::<LE>()
            .map_err(|err| Error::InvalidOther(err.to_string()))?;

        let mut properties = HashMap::with_capacity(num_properties.try_into().unwrap());

        for _ in 0..num_properties {
            let property = Property::from_reader(reader)?;
            properties.insert(property.property_name, property.property_value);
        }
        Ok(Self { properties })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decompression() {
        let input: [i16; 8] = [4, 3, 10, 1, 10, 30, 20, 40];
        let mut input_array = ndarray::Array2::from_shape_vec((4, 2), input.to_vec()).unwrap();
        let calculated_output = PixelDataSupported::decompress_diffs(input_array.view_mut());
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
        let calculated_output = PixelDataSupported::parse_lookup(test);
        let output: Vec<u8> = vec![2, 1, 1, 1, 4, 4, 1, 1, 4, 8, 2, 1, 1, 2, 2, 1, 1, 4, 4, 1];
        assert_eq!(output, calculated_output);
    }
}
