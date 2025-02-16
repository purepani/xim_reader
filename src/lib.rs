use pyo3::prelude::*;
mod reader;

/// Formats the sum of two numbers as string.
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

/// A Python module implemented in Rust.
#[pymodule]
fn xim_reader(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<reader::XIMImage>();
    Ok(())
}
