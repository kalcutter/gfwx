//  Good, Fast Wavelet Codec "GFWX" v1
//  ----------------------------------
//  December 1, 2015 [patched on Oct 20, 2017]
//  Author: Graham Fyffe <gfyffe@gmail.com> or <fyffe@google.com>, and Google, Inc.
//  Website: www.gfwx.org
//  Features:
//  - FAST
//  - compression ratio similar to JPEG 2000
//  - under 1000 lines of code, with no external libraries
//  - 100% lossless at max quality
//  - low quality looks interpolated instead of blocky
//  - progressive decoding with optional downsampling
//  - supports uint8_t, int8_t, uint16_t, int16_t
//  - supports 1 to 65536 interleaved channels
//  - supports 1 to 65536 non-interleaved layers
//  - optional Bayer mode to compress Bayer data more
//  - optional chroma downsampling, even in Bayer mode
//  - optional user-programmable color/channel transform
//  - optional slightly less fast mode to compress more
//  - imageData can be any class with a pointer-like interface
//  - thoroughly tested using several pictures of cats
//
//  GFWX is released under the 3-clause BSD license:
//
//  Copyright (c) 2015, University of Southern California. All rights reserved. Redistribution and use in source and binary forms,
//  with or without modification, are permitted provided that the following conditions are met:
//
//  1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
//
//  2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in
//     the documentation and/or other materials provided with the distribution.
//
//  3. Neither the name of the organization nor the names of its contributors may be used to endorse or promote products derived from
//     this software without specific prior written permission.
//
//  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
//  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
//  COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
//  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
//  HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
//  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>
#if defined(_OPENMP) && defined(_MSC_VER)
#define OMP_PARALLEL_FOR(X) __pragma(omp parallel for schedule(dynamic, X))
#elif defined(_OPENMP)
#include <omp.h>
#define STR(X) #X
#define STRINGIFY(X) STR(X)
#define TWO_ARGUMENTS(X,Y,Z) X(Y, Z)
#define OMP_PARALLEL_FOR(X) _Pragma(STRINGIFY(TWO_ARGUMENTS(omp parallel for schedule, dynamic, X)))
#else
#define OMP_PARALLEL_FOR(X)
#endif

namespace GFWX
{
	enum
	{
		QualityMax = 1024,		// compress with QualityMax for 100% lossless, or less than QualityMax for lossy
		ThreadIterations = 64,	// OMP settings tuned on my machine with large images
		BitDepthAuto = 0, BlockDefault = 7, BlockMax = 30,
		FilterLinear = 0, FilterCubic = 1, QuantizationScalar = 0, EncoderTurbo = 0, EncoderFast = 1, EncoderContextual = 2,
		IntentGeneric = 0, IntentMono = 1, IntentBayerRGGB = 2, IntentBayerBGGR = 3, IntentBayerGRBG = 4, IntentBayerGBRG = 5, IntentBayerGeneric = 6,
		IntentRGB = 7, IntentRGBA = 8, IntentRGBApremult = 9, IntentBGR = 10, IntentBGRA = 11, IntentBGRApremult = 12, IntentCMYK = 13,
		ResultOk = 0, ErrorOverflow = -1, ErrorMalformed = -2, ErrorTypeMismatch = -3
	};

	struct Header	// use the empty constructor to fetch headers before decompressing, and use the parameterized constructor when compressing
	{
		int sizex, sizey, layers, channels, bitDepth, quality, chromaScale, blockSize, filter, quantization, encoder, intent, version, isSigned;
		Header() {}
		Header(int sizex, int sizey, int layers, int channels, int bitDepth,
			int quality, int chromaScale, int blockSize, int filter, int quantization, int encoder, int intent)
			: sizex(sizex), sizey(sizey), layers(layers), channels(channels), bitDepth(bitDepth), quality(std::max(1, std::min(int(QualityMax), quality))),
			chromaScale(std::max(1, std::min(256, chromaScale))), blockSize(std::min(30, std::max(2, blockSize))), filter(std::min(255, filter)),
			quantization(std::min(255, quantization)), encoder(std::min(255, encoder)), intent(std::min(255, intent)) {}
    size_t bufferSize() const
    {
      size_t const part1 = static_cast<size_t>(sizex) * sizey;
      size_t const part2 = static_cast<size_t>(channels) * layers * ((bitDepth + 7) / 8);
      return std::log(part1) + std::log(part2) > std::log(std::numeric_limits<size_t>::max() - 1) ? 0 : part1 * part2;
    }
	};

	template<typename T> struct Image	// handy wrapper for 2D image data
	{
		T * data;
		int sizex, sizey;
		Image(T * data, int sizex, int sizey) : data(data), sizex(sizex), sizey(sizey) {}
		T * operator[] (int y) { return data + static_cast<size_t>(y) * sizex; }
	};

	struct Bits	// handy wrapper for treating an array of unsigned ints as a bit stream
	{
		uint32_t * buffer, * bufferEnd;
		uint32_t writeCache;
		int indexBits;	// -1 indicates buffer overflow
		Bits(uint32_t * buffer, uint32_t * bufferEnd) : buffer(buffer), bufferEnd(bufferEnd), writeCache(0), indexBits(0) {}
		uint32_t getBits(int bits)
		{
			int newBits = indexBits + bits;
			if (buffer == bufferEnd)
				return indexBits = -1;	// signify overflow
			uint32_t x = *buffer << indexBits;
			if (newBits >= 32)
			{
				++ buffer;
				if ((newBits -= 32) > 0)
				{
					if (buffer == bufferEnd)
						return indexBits = -1;	// signify overflow
					x |= *buffer >> (32 - indexBits);
				}
			}
			indexBits = newBits;
			return x >> (32 - bits);
		}
		void putBits(uint32_t x, int bits)
		{
			int newBits = indexBits + bits;
			if (buffer == bufferEnd)
				newBits = -1;	// signify overflow
			else if (newBits < 32)
				(writeCache <<= bits) |= x;
			else if (bits == 32 && newBits == 32)
			{
				newBits = 0;
				*(buffer ++) = x;
			}
			else
			{
				newBits -= 32;
				*(buffer ++) = (writeCache << (bits - newBits)) | (x >> newBits);
				writeCache = x;
			}
			indexBits = newBits;
		}
		uint32_t getZeros(uint32_t maxZeros)
		{
			int newBits = indexBits;
			if (buffer == bufferEnd)
				return indexBits = -1;	// signify overflow
			uint32_t b = *buffer;
			uint32_t x = 0;
			while (true)
			{
				if (newBits == 31)
				{
					++ buffer;
					if ((b & 1u) || (++ x == maxZeros))
					{
						indexBits = 0;
						return x;
					}
					if (buffer == bufferEnd)
						return indexBits = -1;	// signify overflow
					b = *buffer;
					newBits = 0;
					continue;
				}
				if (((b << newBits) & (1u << 31)) || (++ x == maxZeros))
				{
					indexBits = newBits + 1;
					return x;
				}
				++ newBits;
			}
		}
		void flushWriteWord()	// [NOTE] does not clear overflow
		{
			putBits(0, (32 - indexBits) % 32);
		}
		void flushReadWord()	// [NOTE] does not clear overflow
		{
			if (indexBits <= 0)
				return;
			++ buffer;
			indexBits = 0;
		}
	};

	template<int pot> void unsignedCode(uint32_t x, Bits & stream)	// limited length power-of-two Golomb-Rice code
	{
		uint32_t const y = x >> (pot);
		if (y >= 12)
		{
			stream.putBits(0, 12);	// escape to larger code
			unsignedCode<pot < 20 ? pot + 4 : 24>(x - (12 << (pot)), stream);
		}
		else
			stream.putBits((1 << (pot)) | (x & ~(~0u << (pot))), y + 1 + pot);	// encode x / 2^pot in unary followed by x % 2^pot in binary
	}

	template<int pot> uint32_t unsignedDecode(Bits & stream)
	{
		uint32_t x = stream.getZeros(12);
    int const p = pot < 24 ? pot : 24;  // actual pot. The max 108 below is to prevent unlimited recursion in malformed files, yet admit 2^32 - 1.
		return (pot < 108 && x == 12) ? (12 << p) + unsignedDecode<pot < 108 ? pot + 4 : 108>(stream) : p ? (x << p) + stream.getBits(p) : x;
	}

	template<int pot> void interleavedCode(int x, Bits & stream)
	{
		unsignedCode<pot>(x <= 0 ? -2 * x : 2 * x - 1, stream);	// interleave positive and negative values
	}

	template<int pot> int interleavedDecode(Bits & stream)
	{
		int const x = unsignedDecode<pot>(stream);
		return (x & 1) ? (x + 1) / 2 : -x / 2;
	}

	template<int pot> void signedCode(int x, Bits & stream)
	{
		unsignedCode<pot>(abs(x), stream);
		if (x)
			stream.putBits(x > 0 ? 1 : 0, 1);
	}

	template<int pot> int signedDecode(Bits & stream)
	{
		int x = unsignedDecode<pot>(stream);
		return x ? stream.getBits(1) ? x : -x : 0;
	}

	template<typename T> T median(T a, T b, T c)
	{
		return a < b ? c > b ? b : c < a ? a : c : c > a ? a : c < b ? b : c;
	}

	template<typename T> T roundFraction(T num, T denom)
	{
		return num < 0 ? (num - denom / 2) / denom : (num + denom / 2) / denom;
	}

	template<typename T> T cubic(T c0, T c1, T c2, T c3)
	{
		return median(T(roundFraction((-c0 + 9 * (c1 + c2) - c3), 16)), c1, c2);
	}

	template<typename T> void lift(Image<T> & image, int x0, int y0, int x1, int y1, int step, int filter)
	{
		int const sizex = x1 - x0;
		int const sizey = y1 - y0;
		while (step < sizex || step < sizey)
		{
			if (step < sizex)	// horizontal lifting
			{
				OMP_PARALLEL_FOR(ThreadIterations)
				for (int y = 0; y < sizey; y += step)
				{
					int x;
					T * base = &image[y0 + y][x0], * base1 = base - step, * base2 = base + step, * base3 = base + step * 3;
					if (filter == FilterCubic)
					{
						T c0 = *base, c1 = *base, c2 = step * 2 < sizex ? base[step * 2] : *base, c3;
						for (x = step; x < sizex - step * 3; x += step * 2, c0 = c1, c1 = c2, c2 = c3)
							base[x] -= cubic(c0, c1, c2, c3 = base3[x]);
						for (; x < sizex; x += step * 2, c0 = c1, c1 = c2)
							base[x] -= cubic(c0, c1, c2, c2);
						T g0 = base[step], g1 = base[step], g2 = step * 3 < sizex ? base[step * 3] : base[step], g3;
						for (x = step * 2; x < sizex - step * 3; x += step * 2, g0 = g1, g1 = g2, g2 = g3)
							base[x] += cubic(g0, g1, g2, g3 = base3[x]) / 2;
						for (; x < sizex; x += step * 2, g0 = g1, g1 = g2)
							base[x] += cubic(g0, g1, g2, g2) / 2;
					}
					else
					{
						for (x = step; x < sizex - step; x += step * 2)
							base[x] -= (base1[x] + base2[x]) / 2;
						if (x < sizex)
							base[x] -= base1[x];
						for (x = step * 2; x < sizex - step; x += step * 2)
							base[x] += (base1[x] + base2[x]) / 4;
						if (x < sizex)
							base[x] += base1[x] / 2;
					}
				}
			}
			if (step < sizey)	// vertical lifting
			{
				OMP_PARALLEL_FOR(ThreadIterations)
				for (int y = step; y < sizey; y += step * 2)
				{
					T * const base = &image[y0 + y][x0];
					T const * const c1base = &image[y0 + y - step][x0], * const c2base = y + step < sizey ? &image[y0 + y + step][x0] : c1base;
					if (filter == FilterCubic)
					{
						T const * const c0base = y - step * 3 >= 0 ? &image[y0 + y - step * 3][x0] : c1base;
						T const * const c3base = y + step * 3 < sizey ? &image[y0 + y + step * 3][x0] : c2base;
						for (int x = 0; x < sizex; x += step)
							base[x] -= cubic(c0base[x], c1base[x], c2base[x], c3base[x]);
					}
					else for (int x = 0; x < sizex; x += step)
						base[x] -= (c1base[x] + c2base[x]) / 2;
				}
				OMP_PARALLEL_FOR(ThreadIterations)
				for (int y = step * 2; y < sizey; y += step * 2)
				{
					T * const base = &image[y0 + y][x0];
					T const * const g1base = &image[y0 + y - step][x0], * const g2base = y + step < sizey ? &image[y0 + y + step][x0] : g1base;
					if (filter == FilterCubic)
					{
						T const * const g0base = y - step * 3 >= 0 ? &image[y0 + y - step * 3][x0] : g1base;
						T const * const g3base = y + step * 3 < sizey ? &image[y0 + y + step * 3][x0] : g2base;
						for (int x = 0; x < sizex; x += step)
							base[x] += cubic(g0base[x], g1base[x], g2base[x], g3base[x]) / 2;
					}
					else for (int x = 0; x < sizex; x += step)
						base[x] += (g1base[x] + g2base[x]) / 4;
				}
			}
			step *= 2;
		}
	}

	template<typename T> void unlift(Image<T> & image, int x0, int y0, int x1, int y1, int minStep, int filter)
	{
		int const sizex = x1 - x0;
		int const sizey = y1 - y0;
		int step = minStep;
		while (step * 2 < sizex || step * 2 < sizey)
			step *= 2;
		while (step >= minStep)
		{
			if (step < sizey)	// vertical unlifting
			{
				OMP_PARALLEL_FOR(ThreadIterations)
				for (int y = step * 2; y < sizey; y += step * 2)
				{
					T * const base = &image[y0 + y][x0];
					T const * const g1base = &image[y0 + y - step][x0], * const g2base = y + step < sizey ? &image[y0 + y + step][x0] : g1base;
					if (filter == FilterCubic)
					{
						T const * const g0base = y - step * 3 >= 0 ? &image[y0 + y - step * 3][x0] : g1base;
						T const * const g3base = y + step * 3 < sizey ? &image[y0 + y + step * 3][x0] : g2base;
						for (int x = 0; x < sizex; x += step)
							base[x] -= cubic(g0base[x], g1base[x], g2base[x], g3base[x]) / 2;
					}
					else for (int x = 0; x < sizex; x += step)
						base[x] -= (g1base[x] + g2base[x]) / 4;
				}
				OMP_PARALLEL_FOR(ThreadIterations)
				for (int y = step; y < sizey; y += step * 2)
				{
					T * const base = &image[y0 + y][x0];
					T const * const c1base = &image[y0 + y - step][x0], * const c2base = y + step < sizey ? &image[y0 + y + step][x0] : c1base;
					if (filter == FilterCubic)
					{
						T const * const c0base = y - step * 3 >= 0 ? &image[y0 + y - step * 3][x0] : c1base;
						T const * const c3base = y + step * 3 < sizey ? &image[y0 + y + step * 3][x0] : c2base;
						for (int x = 0; x < sizex; x += step)
							base[x] += cubic(c0base[x], c1base[x], c2base[x], c3base[x]);
					}
					else for (int x = 0; x < sizex; x += step)
						base[x] += (c1base[x] + c2base[x]) / 2;
				}
			}
			if (step < sizex)	// horizontal unlifting
			{
				OMP_PARALLEL_FOR(ThreadIterations)
				for (int y = 0; y < sizey; y += step)
				{
					int x;
					T * base = &image[y0 + y][x0], * base1 = base - step, * base2 = base + step, * base3 = base + step * 3;
					if (filter == FilterCubic)
					{
						T g0 = base[step], g1 = base[step], g2 = step * 3 < sizex ? base[step * 3] : base[step], g3;
						for (x = step * 2; x < sizex - step * 3; x += step * 2, g0 = g1, g1 = g2, g2 = g3)
							base[x] -= cubic(g0, g1, g2, g3 = base3[x]) / 2;
						for (; x < sizex; x += step * 2, g0 = g1, g1 = g2)
							base[x] -= cubic(g0, g1, g2, g2) / 2;
						T c0 = *base, c1 = *base, c2 = step * 2 < sizex ? base[step * 2] : *base, c3;
						for (x = step; x < sizex - step * 3; x += step * 2, c0 = c1, c1 = c2, c2 = c3)
							base[x] += cubic(c0, c1, c2, c3 = base3[x]);
						for (; x < sizex; x += step * 2, c0 = c1, c1 = c2)
							base[x] += cubic(c0, c1, c2, c2);
					}
					else
					{
						for (x = step * 2; x < sizex - step; x += step * 2)
							base[x] -= (base1[x] + base2[x]) / 4;
						if (x < sizex)
							base[x] -= base1[x] / 2;
						for (x = step; x < sizex - step; x += step * 2)
							base[x] += (base1[x] + base2[x]) / 2;
						if (x < sizex)
							base[x] += base1[x];
					}
				}
			}
			step /= 2;
		}
	}

	template<typename T, bool dequantize> void quantize(Image<T> & image, int x0, int y0, int x1, int y1, int step, int quality, int minQ, int maxQ)
	{
		typedef typename std::conditional<sizeof(T) < 4, int32_t, int64_t>::type aux;
		int const sizex = x1 - x0;
		int const sizey = y1 - y0;
		int skip = step;
		while (skip < sizex && skip < sizey)
		{
			int const q = std::max(std::max(1, minQ), quality);
			if (q >= maxQ) break;
			OMP_PARALLEL_FOR(ThreadIterations)
			for (int y = 0; y < sizey; y += skip)
			{
				T * base = &image[y0 + y][x0];
				int const xStep = (y & skip) ? skip : skip * 2;
				for (int x = xStep - skip; x < sizex; x += xStep)	// [NOTE] arranged so that (x | y) & skip == 1
					base[x] = dequantize ? (aux(base[x]) * maxQ + (base[x] < 0 ? -maxQ / 2 : base[x] > 0 ? maxQ / 2 : 0)) / q : aux(base[x]) * q / maxQ;
			}
			skip *= 2;
			quality = std::min(maxQ, quality * 2);	// [MAGIC] This approximates the JPEG 2000 baseline quantizer
		}
	}

	template<typename T> T square(T t)
	{
		return t * t;
	}

	inline void addContext(int x, int w, uint32_t & sum, uint32_t & sum2, uint32_t & count)
	{
		sum += uint32_t(x = abs(x)) * w;
		sum2 += square(std::min(uint32_t(x), 4096u)) * w;	// [MAGIC] avoid overflow in last line of getContext
		count += w;
	}

	template<typename T> std::pair<uint32_t, uint32_t> getContext(Image<T> & image, int x0, int y0, int x1, int y1, int x, int y, int skip)
	{
		int px = x0 + (x & ~(skip * 2)) + (x & skip);
		if (px >= x1)
			px -= skip * 2;
		int py = y0 + (y & ~(skip * 2)) + (y & skip);
		if (py >= y1)
			py -= skip * 2;
		uint32_t count = 0, sum = 0, sum2 = 0;
		addContext(abs(image[py][px]), 2, sum, sum2, count);	// ancestor
		if ((y & skip) && (x | skip) < x1 - x0)
		{
			addContext(image[y0 + y - skip][x0 + (x | skip)], 2, sum, sum2, count);	// upper sibling
			if (x & skip)
				addContext(image[y0 + y][x0 + x - skip], 2, sum, sum2, count);	// left sibling
		}
		if (y >= skip * 2 && x >= skip * 2)	// neighbors
		{
			addContext(image[y0 + y - skip * 2][x0 + x], 4, sum, sum2, count);
			addContext(image[y0 + y][x0 + x - skip * 2], 4, sum, sum2, count);
			addContext(image[y0 + y - skip * 2][x0 + x - skip * 2], 2, sum, sum2, count);
			if (x + skip * 2 < x1 - x0)
				addContext(image[y0 + y - skip * 2][x0 + x + skip * 2], 2, sum, sum2, count);
			if (y >= skip * 4 && x >= skip * 4)
			{
				addContext(image[y0 + y - skip * 4][x0 + x], 2, sum, sum2, count);
				addContext(image[y0 + y][x0 + x - skip * 4], 2, sum, sum2, count);
				addContext(image[y0 + y - skip * 4][x0 + x - skip * 4], 1, sum, sum2, count);
				if (x + skip * 4 < x1 - x0)
					addContext(image[y0 + y - skip * 4][x0 + x + skip * 4], 1, sum, sum2, count);
			}
		}
		return std::make_pair((sum * 16u + count / 2u) / count, (sum2 * 16u + count / 2u) / count);	// set sums relative to 16 count
	}

	template<typename T> void encode(Image<T> & image, Bits & stream, int x0, int y0, int x1, int y1, int step, int scheme, int q, bool hasDC, bool isChroma)
	{
		int const sizex = x1 - x0;
		int const sizey = y1 - y0;
		if (hasDC && sizex > 0 && sizey > 0)
			signedCode<4>(image[y0][x0], stream);
		std::pair<uint32_t, uint32_t> context(0, 0);
		int run = 0, runCoder = (scheme == EncoderTurbo ? (!q || (step < 2048 && q * step < 2048)) ? 1 : 0 : 0);  // avoid overflow checking q * step < 2048
		for (int y = 0; y < sizey; y += step)
		{
			T * base = &image[y0 + y][x0];
			int const xStep = (y & step) ? step : step * 2;
			for (int x = xStep - step; x < sizex; x += xStep)	// [NOTE] arranged so that (x | y) & step == 1
			{
				T s = base[x];
				if (runCoder && !s)	// run
					++ run;
				else
				{
					if (scheme == EncoderTurbo)
					{
						if (runCoder)	// break the run
						{
							unsignedCode<1>(run, stream);
							run = 0;
							interleavedCode<1>(s < 0 ? s + 1 : s, stream);	// s can't be zero, so shift negatives by 1
						}
						else
							interleavedCode<1>(s, stream);
						continue;
					}
					if (runCoder)	// break the run
					{
						runCoder == 1 ? unsignedCode<1>(run, stream) : runCoder == 2 ? unsignedCode<2>(run, stream)
							: runCoder == 3 ? unsignedCode<3>(run, stream) : unsignedCode<4>(run, stream);
						run = 0;
						if (s < 0)
							++ s;	// s can't be zero, so shift negatives by 1
					}
					if (scheme == EncoderContextual)
						context = getContext(image, x0, y0, x1, y1, x, y, step);
					uint32_t const sumSq = square(context.first);
					if (sumSq < 2u * context.second + (isChroma ? 250u : 100u))
						interleavedCode<0>(s, stream);
					else if (sumSq < 2u * context.second + 950u)
						interleavedCode<1>(s, stream);
					else if (sumSq < 3u * context.second + 3000u)
					{
						if (sumSq < 5u * context.second + 400u)
							signedCode<1>(s, stream);
						else
							interleavedCode<2>(s, stream);
					}
					else if (sumSq < 3u * context.second + 12000u)
					{
						if (sumSq < 5u * context.second + 3000u)
							signedCode<2>(s, stream);
						else
							interleavedCode<3>(s, stream);
					}
					else if (sumSq < 4u * context.second + 44000u)
					{
						if (sumSq < 6u * context.second + 12000u)
							signedCode<3>(s, stream);
						else
							interleavedCode<4>(s, stream);
					}
					else
						signedCode<4>(s, stream);
					if (scheme == EncoderFast)	// use decaying first and second moment
					{
						uint32_t const t = abs(s);
						context = std::make_pair(((context.first * 15u + 7u) >> 4) + t, ((context.second * 15u + 7u) >> 4) + square(std::min(t, 4096u)));
						if (!s == !runCoder)
							runCoder = context.first < 1 ? 4 : context.first < 2 ? 3 : context.first < 4 ? 2 : context.first < 8 ? 1 : 0;
					}
					else if (!s == !runCoder)
						runCoder = q == 1024 ? context.first < 2u ? 1 : 0 : (context.first < 4u && context.second < 2u) ? 4 : (context.first < 8u
						&& context.second < 4u) ? 3 : (2u * sumSq < 3u * context.second + 48u) ? 2 : (2u * sumSq < 5u * context.second + 32u) ? 1 : 0;
				}
			}
		}
		if (run)	// flush run
			runCoder == 1 ? unsignedCode<1>(run, stream) : runCoder == 2 ? unsignedCode<2>(run, stream)
				: runCoder == 3 ? unsignedCode<3>(run, stream) : unsignedCode<4>(run, stream);
	}

	template<typename T> void decode(Image<T> & image, Bits & stream, int x0, int y0, int x1, int y1, int step, int scheme, int q, bool hasDC, bool isChroma)
	{
		int const sizex = x1 - x0;
		int const sizey = y1 - y0;
		if (hasDC && sizex > 0 && sizey > 0)
			image[y0][x0] = signedDecode<4>(stream);
		std::pair<uint32_t, uint32_t> context(0, 0);
		int run = -1, runCoder = (scheme == EncoderTurbo ? (!q || (step < 2048 && q * step < 2048)) ? 1 : 0 : 0);  // avoid overflow checking q * step < 2048
		for (int y = 0; y < sizey; y += step)
		{
			T * base = &image[y0 + y][x0];
			int const xStep = (y & step) ? step : step * 2;
			for (int x = xStep - step; x < sizex; x += xStep)	// [NOTE] arranged so that (x | y) & step == 1
			{
				T s = 0;
				if (runCoder && run == -1)
					run = runCoder == 1 ? unsignedDecode<1>(stream) : runCoder == 2 ? unsignedDecode<2>(stream)
						: runCoder == 3 ? unsignedDecode<3>(stream) : unsignedDecode<4>(stream);
				if (run > 0)
					-- run;	// consume a zero
				else
				{
					if (scheme == EncoderTurbo)
						s = interleavedDecode<1>(stream);
					else
					{
						if (scheme == EncoderContextual)
							context = getContext(image, x0, y0, x1, y1, x, y, step);
						uint32_t const sumSq = square(context.first);
						if (sumSq < 2u * context.second + (isChroma ? 250u : 100u))
							s = interleavedDecode<0>(stream);
						else if (sumSq < 2u * context.second + 950u)
							s = interleavedDecode<1>(stream);
						else if (sumSq < 3u * context.second + 3000u)
						{
							if (sumSq < 5u * context.second + 400u)
								s = signedDecode<1>(stream);
							else
								s = interleavedDecode<2>(stream);
						}
						else if (sumSq < 3u * context.second + 12000u)
						{
							if (sumSq < 5u * context.second + 3000u)
								s = signedDecode<2>(stream);
							else
								s = interleavedDecode<3>(stream);
						}
						else if (sumSq < 4u * context.second + 44000u)
						{
							if (sumSq < 6u * context.second + 12000u)
								s = signedDecode<3>(stream);
							else
								s = interleavedDecode<4>(stream);
						}
						else
							s = signedDecode<4>(stream);
						if (scheme == EncoderFast)	// use decaying first and second moment
						{
							uint32_t const t = abs(s);
							context = std::make_pair(((context.first * 15u + 7u) >> 4) + t, ((context.second * 15u + 7u) >> 4) + square(std::min(t, 4096u)));
							if (!s == !runCoder)
								runCoder = context.first < 1 ? 4 : context.first < 2 ? 3 : context.first < 4 ? 2 : context.first < 8 ? 1 : 0;
						}
						else if (!s == !runCoder)
							runCoder = q == 1024 ? context.first < 2u ? 1 : 0 : (context.first < 4u && context.second < 2u) ? 4 : (context.first < 8u
							&& context.second < 4u) ? 3 : (2u * sumSq < 3u * context.second + 48u) ? 2 : (2u * sumSq < 5u * context.second + 32u) ? 1 : 0;
					}
					if (run == 0 && s <= 0)
						-- s;	// s can't be zero, so shift negatives by 1
					run = -1;
				}
				base[x] = s;
			}
		}
	}

	template<typename T> void shiftVector(T * data, int shift, int count)
	{
		OMP_PARALLEL_FOR(ThreadIterations * ThreadIterations)
		for (int i = 0; i < count; ++ i)
			data[i] >>= shift;
	}

	template<typename I, typename A> void transformTerm(int const * & pc, A * destination, A const * auxData, int const bufferSize,
		I const & imageData, Header const & header, std::vector<int> const & isChroma, int boost)
	{
		while (*pc >= 0)
		{
			int const c = *(pc ++);
			A const factor = *(pc ++);
			if (isChroma[c] == -1)
			{
				auto layer = imageData + ((c / header.channels) * bufferSize * header.channels + c % header.channels);
				A const boostFactor = boost * factor;
				OMP_PARALLEL_FOR(ThreadIterations * ThreadIterations)
				for (int i = 0; i < bufferSize; ++ i)
					destination[i] += layer[i * header.channels] * boostFactor;
			}
			else
			{
				A const * auxDataC = auxData + c * bufferSize;
				OMP_PARALLEL_FOR(ThreadIterations * ThreadIterations)
				for (int i = 0; i < bufferSize; ++ i)
					destination[i] += auxDataC[i] * factor;
			}
		}
		A const denom = *((++ pc) ++);
		if (denom == 2)
			shiftVector(destination, 1, bufferSize);
		else if (denom == 4)
			shiftVector(destination, 2, bufferSize);
		else if (denom == 8)
			shiftVector(destination, 3, bufferSize);
		else if (denom > 1)	// [NOTE] disallow non-positive denominators
		{
			OMP_PARALLEL_FOR(ThreadIterations * ThreadIterations)
			for (int i = 0; i < bufferSize; ++ i)
				destination[i] /= denom;
		}
	}

	// GFWX_TRANSFORM_UYV implements YUV (actually UYV) as R -= G (chroma); B -= G (chroma); G += (R + B) / 4 (luma)
	#define GFWX_TRANSFORM_UYV { 0, 1, -1, -1, 1, 1, 2, 1, -1, -1, 1, 1, 1, 0, 1, 2, 1, -1, 4, 0, -1 }
	// GFWX_TRANSFORM_A710 implements A710 as R -= G (chroma); B -= (G * 2 + R) / 2 (chroma); G += (B * 2 + R * 3) / 8 (luma)
	#define GFWX_TRANSFORM_A710_BGR { 2, 1, -1, -1, 1, 1, 0, 1, -2, 2, -1, -1, 2, 1, 1, 0, 2, 2, 3, -1, 8, 0, -1 }
	#define GFWX_TRANSFORM_A710_RGB { 0, 1, -1, -1, 1, 1, 2, 1, -2, 0, -1, -1, 2, 1, 1, 2, 2, 0, 3, -1, 8, 0, -1 }

	template<typename I> ptrdiff_t compress(I const & imageData, Header & header, uint8_t * buffer, size_t size,
		int const * channelTransform, uint8_t * metaData, size_t metaDataSize)
	{
		typedef typename std::remove_reference<decltype(imageData[0])>::type base;
		typedef typename std::conditional<sizeof(base) < 2, int16_t, int32_t>::type aux;
    if (header.sizex > (1 << 30) || header.sizey > (1 << 30))  // [NOTE] current implementation can't go over 2^30
      return ErrorMalformed;
		Bits stream(reinterpret_cast<uint32_t *>(buffer), reinterpret_cast<uint32_t *>(buffer) + size / 4);
		stream.putBits('G' | ('F' << 8) | ('W' << 16) | ('X' << 24), 32);
		stream.putBits(header.version = 1, 32);
		stream.putBits(header.sizex, 32);
		stream.putBits(header.sizey, 32);
		stream.putBits(header.layers - 1, 16);
		stream.putBits(header.channels - 1, 16);
		stream.putBits((header.bitDepth ? header.bitDepth : (header.bitDepth = std::numeric_limits<base>::digits)) - 1, 8);
		stream.putBits(header.isSigned = std::numeric_limits<base>::is_signed ? 1 : 0, 1);
		stream.putBits(header.quality - 1, 10);
		stream.putBits(header.chromaScale - 1, 8);
		stream.putBits(header.blockSize - 2, 5);
		stream.putBits(header.filter, 8);
		stream.putBits(header.quantization, 8);
		stream.putBits(header.encoder, 8);
		stream.putBits(header.intent, 8);
		stream.putBits(int(metaDataSize / 4), 32);
		stream.buffer = std::copy(reinterpret_cast<uint32_t *>(metaData), reinterpret_cast<uint32_t *>(metaData) + metaDataSize / 4, stream.buffer);
		int const bufferSize = header.sizex * header.sizey;
		std::vector<aux> auxData((size_t)header.layers * header.channels * bufferSize, 0);
		std::vector<int> isChroma(header.layers * header.channels, -1);
		int const chromaQuality = std::max(1, (header.quality + header.chromaScale / 2) / header.chromaScale);
		int const boost = header.quality == QualityMax ? 1 : 8;	// [NOTE] due to Cubic lifting max multiplier of 20, boost * 20 must be less than 256
		if (channelTransform)	// run color transform program (and also encode it to the file)
		{
			int const * pc = channelTransform;
			while (*pc >= 0)
			{
				int const c = *(pc ++);
				aux * destination = &auxData[c * bufferSize];
				transformTerm(pc, destination, &auxData[0], bufferSize, imageData, header, isChroma, boost);
				auto layer = imageData + ((c / header.channels) * bufferSize * header.channels + c % header.channels);
				OMP_PARALLEL_FOR(ThreadIterations * ThreadIterations)
				for (int i = 0; i < bufferSize; ++ i)
					destination[i] += layer[i * header.channels] * boost;
				isChroma[c] = *(pc ++);
			}
			for (int const * i = channelTransform; i <= pc; ++ i)
				signedCode<2>(*i, stream);
		}
		else
			signedCode<2>(-1, stream);
		stream.flushWriteWord();
		for (int c = 0; c < header.layers * header.channels; ++ c) if (isChroma[c] == -1)	// copy channels having no transform
		{
			aux * destination = &auxData[c * bufferSize];
			auto layer = imageData + ((c / header.channels) * bufferSize * header.channels + c % header.channels);
			OMP_PARALLEL_FOR(ThreadIterations * ThreadIterations)
			for (int i = 0; i < bufferSize; ++ i)
				destination[i] = layer[i * header.channels] * boost;
			isChroma[c] = 0;
		}
		for (int c = 0; c < header.layers * header.channels; ++ c)	// lift and quantize the channels
		{
			Image<aux> auxImage(&auxData[c * bufferSize], header.sizex, header.sizey);
			lift(auxImage, 0, 0, header.sizex, header.sizey, 1, header.filter);
			if (header.intent >= IntentBayerRGGB && header.intent <= IntentBayerGeneric)
			{
				for (int ox = 0; ox <= 1; ++ ox) for (int oy = 1 - ox; oy <= 1; ++ oy)
					lift(auxImage, ox, oy, header.sizex, header.sizey, 2, header.filter);
				for (int ox = 0; ox <= 1; ++ ox) for (int oy = 0; oy <= 1; ++ oy)
					quantize<aux, false>(auxImage, ox, oy, header.sizex, header.sizey, 2,
						(ox | oy) ? chromaQuality : header.quality, header.quality, QualityMax * boost);
			}
			else
				quantize<aux, false>(auxImage, 0, 0, header.sizex, header.sizey, 1, isChroma[c] ? chromaQuality : header.quality, 0, QualityMax * boost);
		}
		int step = 1;
		while (step * 2 < header.sizex || step * 2 < header.sizey)
			step *= 2;
		for (bool hasDC = true; step >= 1; hasDC = false)
		{
			int64_t const bs = int64_t(step) << header.blockSize;
			int const blockCountX = (header.sizex + bs - 1) / bs;
			int const blockCountY = (header.sizey + bs - 1) / bs;
			int const blockCount = blockCountX * blockCountY * header.layers * header.channels;
			std::vector<Bits> streamBlock(blockCount, Bits(0, 0));
			uint32_t * blockBegin = stream.buffer + blockCount;	// leave space for block sizes
			if (blockBegin >= stream.bufferEnd)
				return ErrorOverflow;
			for (int block = 0; block < blockCount; ++ block)	// partition buffer into temporary regions for each block
				streamBlock[block].buffer = blockBegin + (stream.bufferEnd - blockBegin) * block / blockCount;
			for (int block = 0; block < blockCount; ++ block)
				streamBlock[block].bufferEnd = block + 1 < blockCount ? streamBlock[block + 1].buffer : stream.bufferEnd;
			OMP_PARALLEL_FOR(4)	// [MAGIC] for some reason, 4 is by far the best option here
			for (int block = 0; block < blockCount; ++ block)
			{
				int const bx = block % blockCountX, by = (block / blockCountX) % blockCountY, c = block / (blockCountX * blockCountY);
				Image<aux> auxImage(&auxData[c * bufferSize], header.sizex, header.sizey);
				if (header.intent < IntentBayerRGGB || header.intent > IntentBayerGeneric)
					encode(auxImage, streamBlock[block], bx * bs, by * bs,
					int(std::min((bx + 1) * bs, int64_t(header.sizex))), int(std::min((by + 1) * bs, int64_t(header.sizey))),
					step, header.encoder, isChroma[c] ? chromaQuality : header.quality, hasDC && !bx && !by, isChroma[c] != 0);
				else for (int ox = 0; ox <= 1; ++ ox) for (int oy = 0; oy <= 1; ++ oy)
					encode(auxImage, streamBlock[block], bx * bs + ox, by * bs + oy,
					int(std::min((bx + 1) * bs, int64_t(header.sizex))), int(std::min((by + 1) * bs, int64_t(header.sizey))),
					2 * step, header.encoder, (ox || oy) ? chromaQuality : header.quality, hasDC && !bx && !by, ox || oy);
				streamBlock[block].flushWriteWord();
			}
			for (int block = 0; block < blockCount; ++ block)	// check streamBlocks for overflow
				if (streamBlock[block].indexBits < 0)
					return ErrorOverflow;
			for (int block = 0; block < blockCount; ++ block)	// encode block lengths [NOTE] this 32-bit encoding limits the file size to < 16 GB
				*(stream.buffer ++) = uint32_t(streamBlock[block].buffer - (block ? streamBlock[block - 1].bufferEnd : blockBegin));
			for (int block = 0; block < blockCount; ++ block)	// pack the streamBlock data tightly, by word [NOTE] first block is already packed
				stream.buffer = block ? std::copy(streamBlock[block - 1].bufferEnd, streamBlock[block].buffer, stream.buffer) : streamBlock[0].buffer;
			step /= 2;
		}
		return reinterpret_cast<uint8_t *>(stream.buffer) - buffer;	// return size in bytes
	}

	template<typename I> ptrdiff_t decompress(I const & imageData, Header & header, uint8_t const * data, size_t size, int downsampling, bool test)
	{
		typedef typename std::remove_reference<decltype(imageData[0])>::type base;
		typedef typename std::conditional<sizeof(base) < 2, int16_t, int32_t>::type aux;
		Bits stream(reinterpret_cast<uint32_t *>(const_cast<uint8_t *>(data)), reinterpret_cast<uint32_t *>(const_cast<uint8_t *>(data)) + size / 4);
		if (size < 28)	// at least load the header
			return 28;
		if (stream.getBits(32) != uint32_t('G' | ('F' << 8) | ('W' << 16) | ('X' << 24)))
			return ErrorMalformed;
		header.version = stream.getBits(32);
		header.sizex = stream.getBits(32);
		header.sizey = stream.getBits(32);
		header.layers = stream.getBits(16) + 1;
		header.channels = stream.getBits(16) + 1;
		header.bitDepth = stream.getBits(8) + 1;
		header.isSigned = stream.getBits(1);
		header.quality = stream.getBits(10) + 1;
		header.chromaScale = stream.getBits(8) + 1;
		header.blockSize = stream.getBits(5) + 2;
		header.filter = stream.getBits(8);
		header.quantization = stream.getBits(8);
		header.encoder = stream.getBits(8);
		header.intent = stream.getBits(8);
    if (header.sizex < 0 || header.sizex > (1 << 30) || header.sizey < 0 || header.sizey > (1 << 30) || header.bufferSize() == 0)
      return ErrorMalformed;  // [NOTE] current implementation can't go over 2^30
		if (!imageData)		// just header
			return ResultOk;
		if (header.isSigned != (std::numeric_limits<base>::is_signed ? 1 : 0) || header.bitDepth > std::numeric_limits<base>::digits)
			return ErrorTypeMismatch;	// check for correct buffer type (though doesn't test the buffer size)
		// [NOTE] clients can read metadata themselves by accessing the size (in words) at word[7] and the metadata at word[8+]
		if ((stream.buffer += stream.getBits(32)) >= stream.bufferEnd) // skip metadata
			return reinterpret_cast<uint8_t *>(stream.buffer) - data;	// suggest point of interest to skip metadata
		int const sizexDown = (header.sizex + (1 << downsampling) - 1) >> downsampling, sizeyDown = (header.sizey + (1 << downsampling) - 1) >> downsampling;
		int const bufferSize = sizexDown * sizeyDown;
		std::vector<aux> auxData((size_t)header.layers * header.channels * bufferSize, 0);
		std::vector<int> isChroma(header.layers * header.channels, 0), transformProgram, transformSteps;
		size_t nextPointOfInterest = size + 1024;	// guess next point of interest [NOTE] may be larger than the complete file
		while (true)	// decode color transform program (including isChroma flags)
		{
			transformProgram.push_back(signedDecode<2>(stream));	// channel
			if (transformProgram.back() >= static_cast<int>(isChroma.size()))
          return ErrorMalformed;
			if (transformProgram.back() < 0)
				break;
			transformSteps.push_back(int(transformProgram.size()) - 1);
			while (true)
			{
				if (stream.indexBits < 0)	// test for truncation
					return nextPointOfInterest;	// need more data
				transformProgram.push_back(signedDecode<2>(stream));	// other channel
        if (transformProgram.back() >= static_cast<int>(isChroma.size()))
            return ErrorMalformed;
				if (transformProgram.back() < 0)
					break;
				transformProgram.push_back(signedDecode<2>(stream));	// factor
			}
			transformProgram.push_back(signedDecode<2>(stream));	// denominator
			transformProgram.push_back(signedDecode<2>(stream));	// chroma flag
			isChroma[transformProgram[transformSteps.back()]] = transformProgram.back();
		}
		stream.flushReadWord();
		int const chromaQuality = std::max(1, (header.quality + header.chromaScale / 2) / header.chromaScale);
		int const boost = header.quality == QualityMax ? 1 : 8;	// [NOTE] due to Cubic lifting max multiplier of 20, boost * 20 must be less than 256
		bool isTruncated = false;
		int step = 1;
		while (step * 2 < header.sizex || step * 2 < header.sizey)
			step *= 2;
		for (bool hasDC = true; (step >> downsampling) >= 1; hasDC = false)	// decode just enough coefficients for downsampled image
		{
			int64_t const bs = int64_t(step) << header.blockSize;
			int const blockCountX = int((header.sizex + bs - 1) / bs);
			int const blockCountY = int((header.sizey + bs - 1) / bs);
			int const blockCount = blockCountX * blockCountY * header.layers * header.channels;
			isTruncated = true;
			if (stream.buffer + 1 + blockCount > stream.bufferEnd)	// check for enough buffer to read block sizes
				break;
			std::vector<Bits> streamBlock(blockCount, Bits(0, 0));
			for (int block = 0; block < blockCount; ++ block)	// first, read sizes into bufferEnd pointers
				streamBlock[block].bufferEnd = static_cast<uint32_t *>(0) + *(stream.buffer ++);
			for (int block = 0; block < blockCount; ++ block)	// then convert sizes to true buffer pointers
				streamBlock[block].bufferEnd = (streamBlock[block].buffer = block ? streamBlock[block - 1].bufferEnd : stream.buffer)
												+ (streamBlock[block].bufferEnd - static_cast<uint32_t *>(0));
			stream.buffer = streamBlock[blockCount - 1].bufferEnd;
			nextPointOfInterest = reinterpret_cast<uint8_t *>(stream.buffer + ((step >> downsampling) > 1 ? blockCount * 4 : 0)) - data;
			if (stream.buffer <= stream.bufferEnd)
				isTruncated = false;
			int const stepDown = step >> downsampling;
			int64_t const bsDown = int64_t(stepDown) << header.blockSize;
			OMP_PARALLEL_FOR(4)	// [MAGIC] for some reason, 4 is by far the best option here
			for (int block = 0; block < blockCount; ++ block) if (!test && streamBlock[block].bufferEnd <= stream.bufferEnd)
			{
				int const bx = block % blockCountX, by = (block / blockCountX) % blockCountY, c = block / (blockCountX * blockCountY);
				Image<aux> auxImage(&auxData[c * bufferSize], sizexDown, sizeyDown);
				if (header.intent < IntentBayerRGGB || header.intent > IntentBayerGeneric)
					decode(auxImage, streamBlock[block], int(bx * bsDown), int(by * bsDown),
					int(std::min((bx + 1) * bsDown, int64_t(sizexDown))), int(std::min((by + 1) * bsDown, int64_t(sizeyDown))),
					stepDown, header.encoder, isChroma[c] ? chromaQuality : header.quality, hasDC && !bx && !by, isChroma[c] != 0);
				else for (int ox = 0; ox <= 1; ++ ox) for (int oy = 0; oy <= 1; ++ oy)
					decode(auxImage, streamBlock[block], int(bx * bsDown + ox), int(by * bsDown + oy),
					int(std::min((bx + 1) * bsDown, int64_t(sizexDown))), int(std::min((by + 1) * bsDown, int64_t(sizeyDown))),
					2 * stepDown, header.encoder, (ox || oy) ? chromaQuality : header.quality, hasDC && !bx && !by, ox || oy);
			}
			for (int block = 0; block < blockCount; ++ block)	// check if any blocks ran out of buffer, which should not happen on valid files
				if (streamBlock[block].indexBits < 0)
					return ErrorMalformed;
			step /= 2;
		}
		if (test)
			return isTruncated ? nextPointOfInterest : ResultOk;	// return next point of interest if the data was truncated prior to completing request
		for (int c = 0; c < header.layers * header.channels; ++ c)	// dequantize and unlift the channels
		{
			Image<aux> auxImage(&auxData[c * bufferSize], sizexDown, sizeyDown);
			if (header.intent >= IntentBayerRGGB && header.intent <= IntentBayerGeneric)
			{
				for (int ox = 0; ox <= 1; ++ ox) for (int oy = 0; oy <= 1; ++ oy)
					quantize<aux, true>(auxImage, ox, oy, sizexDown, sizeyDown, 2,
						((ox | oy) ? chromaQuality : header.quality) << downsampling, header.quality, QualityMax * boost);
				for (int ox = 0; ox <= 1; ++ ox) for (int oy = 1 - ox; oy <= 1; ++ oy)
					unlift(auxImage, ox, oy, sizexDown, sizeyDown, 2, header.filter);
			}
			else
				quantize<aux, true>(auxImage, 0, 0, sizexDown, sizeyDown, 1,
					(isChroma[c] ? chromaQuality : header.quality) << downsampling, 0, QualityMax * boost);
			unlift(auxImage, 0, 0, sizexDown, sizeyDown, 1, header.filter);
		}
		for (int s = (int)transformSteps.size() - 1; s >= 0; -- s)	// run color transform program in reverse
		{
			int const * pc = &transformProgram[transformSteps[s]];
			int const c = *(pc ++);
			std::vector<aux> transformTemp(bufferSize, 0);
			transformTerm(pc, &transformTemp[0], &auxData[0], bufferSize, imageData, header, isChroma, boost);
			aux * destination = &auxData[c * bufferSize];
			OMP_PARALLEL_FOR(ThreadIterations * ThreadIterations)
			for (int i = 0; i < bufferSize; ++ i)
				destination[i] -= transformTemp[i];
		}
		for (int c = 0; c < header.layers * header.channels; ++ c)	// copy the channels to the destination buffer
		{
			aux * destination = &auxData[c * bufferSize];
			auto layer = imageData + ((c / header.channels) * bufferSize * header.channels + c % header.channels);
			if (boost == 1)
			{
				OMP_PARALLEL_FOR(ThreadIterations * ThreadIterations)
				for (int i = 0; i < bufferSize; ++ i)
					layer[i * header.channels] = static_cast<base>(std::max(static_cast<aux>(std::numeric_limits<base>::lowest()),
						std::min(static_cast<aux>(std::numeric_limits<base>::max()), static_cast<aux>(destination[i]))));
			}
			else
			{
				OMP_PARALLEL_FOR(ThreadIterations * ThreadIterations)
				for (int i = 0; i < bufferSize; ++ i)
					layer[i * header.channels] = static_cast<base>(std::max(static_cast<aux>(std::numeric_limits<base>::lowest()),
						std::min(static_cast<aux>(std::numeric_limits<base>::max()), static_cast<aux>(destination[i] / boost))));
			}
			if (header.quality < QualityMax && header.intent >= IntentBayerRGGB && header.intent <= IntentBayerGBRG)	// check if Bayer cleanup is required
			{
				int const bayerNoiseThresh = ((QualityMax + header.quality / 2) / header.quality + (QualityMax + chromaQuality / 2) / chromaQuality) * 2;
				Image<aux> auxImage(&auxData[c * bufferSize], sizexDown, sizeyDown);
				OMP_PARALLEL_FOR(ThreadIterations)
				for (int y = 1; y < sizeyDown - 1; ++ y)
					for (int x = 1 + (y + (header.intent == IntentBayerGBRG || header.intent == IntentBayerGRBG ? 1 : 0)) % 2; x < sizexDown - 1; x += 2)
					{
						aux s = auxImage[y][x];
						aux sum = s * 4;
						int count = 4;
						for (int oy = -1; oy <= 1; oy += 2) for (int ox = -1; ox <= 1; ox += 2)
						{
							aux t = auxImage[y + oy][x + ox];
							if (abs(s - t) > bayerNoiseThresh)
								continue;
							sum += t;
							++ count;
						}
						layer[(y * sizexDown + x) * header.channels]
							= static_cast<base>(std::max(static_cast<aux>(std::numeric_limits<base>::lowest()),
								std::min(static_cast<aux>(std::numeric_limits<base>::max()), aux((sum + count / 2) / (count * boost)))));
					}
			}
		}
		return isTruncated ? nextPointOfInterest : ResultOk;	// return next point of interest if the data was truncated prior to completing request
	}
}
