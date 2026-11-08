#!/usr/bin/env python3
"""
Apex-Infer Bit-Level Floating Point & Quantization Formats Converter
Simulates IEEE 754 FP32, FP16, BF16, and FP8 formats (OCP E4M3 and E5M2).
Demonstrates dynamic range, precision limits, subnormal values, and quantization error.
"""

import struct
from typing import Tuple


def float_to_bits(val: float) -> int:
    return struct.unpack('>I', struct.pack('>f', val))[0]


def bits_to_float(bits: int) -> float:
    return struct.unpack('>f', struct.pack('>I', bits))[0]


class FP8E4M3:
    """OCP FP8 E4M3: 1 sign bit, 4 exponent bits (bias=7), 3 mantissa bits."""
    BIAS = 7
    EXP_BITS = 4
    MANT_BITS = 3
    MAX_NORMAL = 448.0
    MIN_SUBNORMAL = 2 ** -9  # 0.001953125

    @classmethod
    def quantize(cls, val: float) -> Tuple[int, float]:
        sign = 1 if val < 0 else 0
        abs_val = abs(val)

        if abs_val > cls.MAX_NORMAL:
            abs_val = cls.MAX_NORMAL

        # Find closest representation
        best_bits = 0
        min_err = float('inf')
        best_val = 0.0

        for exp in range(16):
            for mant in range(8):
                if exp == 15 and mant == 7:
                    continue  # NaN in E4M3
                if exp == 0:
                    dec = (mant / 8.0) * (2 ** (1 - cls.BIAS))
                else:
                    dec = (1.0 + mant / 8.0) * (2 ** (exp - cls.BIAS))

                err = abs(abs_val - dec)
                if err < min_err:
                    min_err = err
                    best_bits = (sign << 7) | (exp << 3) | mant
                    best_val = -dec if sign else dec

        return best_bits, best_val


class FP8E5M2:
    """IEEE FP8 E5M2: 1 sign bit, 5 exponent bits (bias=15), 2 mantissa bits."""
    BIAS = 15
    EXP_BITS = 5
    MANT_BITS = 2
    MAX_NORMAL = 57344.0

    @classmethod
    def quantize(cls, val: float) -> Tuple[int, float]:
        sign = 1 if val < 0 else 0
        abs_val = abs(val)

        if abs_val > cls.MAX_NORMAL:
            abs_val = cls.MAX_NORMAL

        best_bits = 0
        min_err = float('inf')
        best_val = 0.0

        for exp in range(32):
            for mant in range(4):
                if exp == 31:
                    continue  # Inf / NaN
                if exp == 0:
                    dec = (mant / 4.0) * (2 ** (1 - cls.BIAS))
                else:
                    dec = (1.0 + mant / 4.0) * (2 ** (exp - cls.BIAS))

                err = abs(abs_val - dec)
                if err < min_err:
                    min_err = err
                    best_bits = (sign << 7) | (exp << 2) | mant
                    best_val = -dec if sign else dec

        return best_bits, best_val


def print_format_comparison(test_values):
    print("=" * 80)
    print(f"{'Target Value':<14} | {'FP8-E4M3 (Quantized)':<24} | {'FP8-E5M2 (Quantized)':<24}")
    print("=" * 80)
    for v in test_values:
        bits_e4m3, val_e4m3 = FP8E4M3.quantize(v)
        bits_e5m2, val_e5m2 = FP8E5M2.quantize(v)
        print(f"{v:<14.5f} | {val_e4m3:<10.5f} (0x{bits_e4m3:02X}, err={abs(v-val_e4m3):.4f}) | "
              f"{val_e5m2:<10.5f} (0x{bits_e5m2:02X}, err={abs(v-val_e5m2):.4f})")
    print("=" * 80)


def main():
    print("Apex-Infer Numerical Precision Demonstration")
    test_vals = [0.001, 0.05, 0.125, 0.33333, 1.0, 3.14159, 15.75, 128.0, 440.0, 1024.0]
    print_format_comparison(test_vals)


if __name__ == "__main__":
    main()
