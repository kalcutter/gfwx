GFWX: Good, Fast Wavelet Codec
==============================

INFORMATION
-----------
GFWX achieves compression ratios similar to JPEG 2000, but compresses and decompresses several times faster. It was developed to help manage the large amount of video data produced by the New Dimensions in Testimony project. This required real-time, lossy compression of Bayer patterned raw images and none of the existing formats fit the bill. Wavelet image compression is a popular paradigm for lossy and lossless image coding, but the arithmetic encoding typically employed to encode wavelet coefficients can be time consuming. GFWX uses a simple and fast entropy encoder based on limited-length Golomb-Rice codes, and uses exclusively integer arithmetic throughout the entire pipeline. It also supports lossy compression of Bayer patterned data without demosaicing, which is vital for fast encoding of imagery from machine vision cameras.

USING GFWX
----------
GFWX is released under the 3-clause BSD license. Do you need a fast, lightweight, lossy or lossless format for your raw image sensor data? Consider GFWX! You can read more about it and download the source code at http://www.gfwx.org.

Features
--------
* Lossy and lossless compression
* Progressive decoding with optional downsampling
* Fast, simple, optionally multithreaded C++11 implementation
* Supports 8-bit or 16-bit data (or anything in between), signed or unsigned
* Stores up to 65536 channels (not limited to RGBA)
* Stores up to 65536 layers or frames
* Bayer mode to improve compression on raw camera sensor data
* Chroma downsampling option, even in Bayer mode
* Programmable lossless color / channel / layer transform (not limited to YUV or whatever)
* Optionally stores an arbitrary metadata block

ACKNOWLEDGEMENTS
----------------
GFWX was developed by Graham Fyffe. This work was sponsored by the U.S. Army Research Laboratory (ARL) under contract number W911NF-14-D-0005, and the USC Shoah Foundation.
