"""
Utilities to serialize and deserialize numpy arrays
"""

import numpy as np
import struct

ENDIAN = "!" # big endian for network order

DT_FLOAT64 = 0
DT_INT64   = 1

def _code_to_wire_dtype(code: int) -> np.dtype:
    if code == DT_FLOAT64:
        return np.dtype(">f8" if ENDIAN in "!>" else "<f8")
    elif code == DT_INT64:
        return np.dtype(">i8" if ENDIAN in "!>" else "<i8")
    raise ValueError(f"Unknown dtype code, {code}")

def _dtype_to_code_and_wire(dt: np.dtype) -> tuple[int, np.dtype]:
    dt = np.dtype(dt)
    if dt == np.float64:
        return DT_FLOAT64, _code_to_wire_dtype(DT_FLOAT64)
    elif dt == np.int64:
        return DT_INT64, _code_to_wire_dtype(DT_INT64)
    raise ValueError("unspported dtype, only np.float64 and np.int64 supported")

def serialize(array: np.ndarray) -> bytes:
    """
    Serialize an ndarray to bytes, only works on c style numpy arrays.

    Byte layout:
    0-7    : dtype code (0=float64, 1=int64)
    8-15   : size (number of elements)
    16-23  : nbytes (total data bytes)
    24-31  : ndim (number of dimensions)
    32..   : strides (ndim * 8-byte signed ints)
    ...    : shape (ndim * 8-byte signed ints)
    ...    : raw data (nbytes)
    """

    if array.flags.c_contiguous is False:
        raise ValueError("Non c contiguous numpy array provided")

    dtype_code, wire_dtype = _dtype_to_code_and_wire(array.dtype)

    ndim = array.ndim
    size = array.size
    nbytes = array.nbytes

    header_fmt = ENDIAN + "qqqq"  # dtype_code, size, nbytes, ndim
    header_bytes = struct.pack(header_fmt, dtype_code, size, nbytes, ndim)

    strides: tuple[int, ...] = array.strides
    shape:   tuple[int, ...] = array.shape
    ss_fmt = ENDIAN + f"{ndim}q{ndim}q"
    ss_bytes = struct.pack(ss_fmt, *strides, *shape)

    data_bytes = array.astype(wire_dtype, copy=False).tobytes(order="C")

    return header_bytes + ss_bytes + data_bytes


def deserialize(b: bytearray | bytes) -> np.ndarray:
    """deserialize an ndarray from bytes"""

    header_fmt = ENDIAN + "qqqq"
    header_size = struct.calcsize(header_fmt)
    if len(b) < header_size:
        raise ValueError("truncated buffer: missing header")

    dtype_code, size, nbytes, ndim = struct.unpack(header_fmt, b[0:header_size])

    qn = ENDIAN + f"{ndim}q"
    qn_size = struct.calcsize(qn)
    strides_start = header_size
    strides_end = strides_start + qn_size
    if len(b) < strides_end:
        raise ValueError("truncated buffer: missing strides")
    strides = struct.unpack_from(qn, b, strides_start)

    shape_start = strides_end
    shape_end = shape_start + qn_size
    if len(b) < shape_end:
        raise ValueError("truncated buffer: missing shape")
    shape = struct.unpack_from(qn, b, shape_start)

    data_start = shape_end
    data_end = data_start + nbytes
    if len(b) < data_end:
        raise ValueError("truncated buffer: missing data")
    data = b[data_start:data_end]

    dtype = _code_to_wire_dtype(dtype_code)
    return np.ndarray(shape=shape, dtype=dtype, strides=strides, buffer=data, order="C")
