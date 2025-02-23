use pyo3::prelude::*;
mod reader;
use pyo3_stub_gen::define_stub_info_gatherer;

/// Formats the sum of two numbers as string.
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

/// A Python module implemented in Rust.
#[pymodule]
fn xim_reader(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let _ = m.add_class::<reader::XIMImage>();
    let _ = m.add_class::<reader::XIMHeader>();
    let _ = m.add_class::<reader::XIMHistogram>();
    let _ = m.add_class::<reader::XIMProperties>();
    Ok(())
}

define_stub_info_gatherer!(stub_info);
