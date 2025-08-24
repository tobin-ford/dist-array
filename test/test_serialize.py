import dist_array
import pytest

import numpy as np
import struct

arr_2d = np.array([[1,2,3,4],[1,1,1,1]], dtype=np.int64)
ENDIAN = "!" # big/network

def make_serial_for(arr: np.ndarray) -> bytes:
    # header
    header = struct.pack(ENDIAN + "qqqq",
                         1 if arr.dtype == np.int64 else 0,
                         arr.size,
                         arr.nbytes,
                         arr.ndim)
    # strides + shape
    ss = struct.pack(ENDIAN + f"{arr.ndim}q{arr.ndim}q", *arr.strides, *arr.shape)
    # payload (big-endian because ENDIAN="!")
    target = np.dtype(">i8") if arr.dtype == np.int64 else np.dtype(">f8")
    data = arr.astype(target, copy=False).tobytes(order="C")
    return header + ss + data

def test_serialize():
    b = dist_array.serialize.serialize(arr_2d)

    header_fmt = ENDIAN + "qqqq"
    header_end = struct.calcsize(header_fmt)

    recv_dtype_code, recv_size, recv_nbytes, recv_ndim = struct.unpack(header_fmt, b[0:header_end])

    ss_format = ENDIAN + f"{recv_ndim}q{recv_ndim}q"
    ss_start = header_end
    ss_end = ss_start + struct.calcsize(ss_format)
    vals = struct.unpack(ss_format, b[ss_start:ss_end])
    recv_strides, recv_shape = vals[:recv_ndim], vals[recv_ndim:]

    data_start = ss_end
    recv_data = b[data_start:]

    restored_arr = np.ndarray(
        shape=recv_shape,
        dtype=">f8" if recv_dtype_code == 0 else ">i8",
        buffer=recv_data,
        strides=recv_strides,
        order="C"
    )

    np.testing.assert_equal(arr_2d, restored_arr)

def test_deserialize():

    arr_2d = np.array([[1,2,3,4],[1,1,1,1]], dtype=np.int64)
    serial = make_serial_for(arr_2d)
    restored = dist_array.serialize.deserialize(serial)

    np.testing.assert_equal(restored, arr_2d)

    assert restored.shape == arr_2d.shape
    assert restored.strides == arr_2d.strides
    # strict dtype check will fail because wire order dtype might not be the same as the native endian type
    assert restored.dtype.newbyteorder('=') == arr_2d.dtype.newbyteorder('=')
