# xim_reader
This tool is a python extension library to read the XIM image format, written in Rust for improved speed. I haven't had a chance to test this with many images, yet, so this library is in an alpha state. This is basically just a parsing library at the moment, so it is better used as a reader to something like the `XIM` class in [pylinac](https://github.com/jrkerns/pylinac).

## Installation

If just using `pip`,
```bash
pip install xim_reader

```
I recommend `uv` as a python projection manager, where
```bash
uv add xim_reader
```
will work.



## Usage
```python
from xim_reader import XIMImage
import numpy as np

path = "/path/to/image.xim"
xim = XIMImage(path)

#Items parsed from header
header = xim.header
print(header.identifier)
print(header.version)
print(header.width)
print(header.height)
print(header.bits_per_pixel)
print(header.bytes_per_pixel)

#A 2D numpy array read-only view with the shape (header.height, header.width) representing the image
pixels: np.ndarray[np.int8] | np.ndarray[np.int16] | np.ndarray[np.int32] | np.ndarray[np.int64] = xim.numpy

#A list where each index represents a bin of an XIM Histogram
histogram: list[int] = xim.histogram

#Properties contained in XIM image. No processing done on these aside from reading them into a dictionary.
properties: dict[str | int | float | list[float] | list[str]] = xim.properties

```

## Building from source
Aside from needing `uv`, the rust compiler must be installed an availible on `PATH`. I have decided to use the 2024 rust edition, meaning at the moment, only the latest rustc version(1.85) is supported. Future versions should compile the package, though. 
After that, running 
```bash
uv build
```
would build the package.

To install the Rust package into the environment with a debug build, run. Refer to the `maturin` documentation for more info.
```bash
maturin develop
```

## Contributing
Feel free to contribute and improve this. Testing this with different data types and compression images would be helpful as well. I currently have only tested with a 32 bit compressed image.
