use pyo3_stub_gen::Result;
fn main() -> Result<()> {
    let stub = xim_reader::stub_info()?;
    stub.generate()?;
    Ok(())
}
