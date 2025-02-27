//! # XIM Reader
//! This crate can read the XIM image format from Varian, powered by Rust for extra speed. The
//! inspiration comes from [pylinac](https://github.com/jrkerns/pylinac), which was referenced and tested against for implementation, and the spec used as a reference can be found [here](https://bitbucket.org/dmoderesearchtools/ximreader/raw/4900d324d5f28f8b6b57752cfbf4282b778a4508/XimReader/xim_readme.pdf).
//! This is currently in an alpha state, as I haven't had a large variation of images to test this
//! with.

use pyo3::prelude::{Bound, PyModule, PyModuleMethods, PyResult, pymodule};
pub mod error;
pub mod reader;
use pyo3_stub_gen::define_stub_info_gatherer;

#[pymodule]
fn xim_reader(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let _ = m.add_class::<reader::XIMImage>();
    let _ = m.add_class::<reader::XIMHeader>();
    let _ = m.add_class::<reader::XIMHistogram>();
    Ok(())
}

define_stub_info_gatherer!(stub_info);
