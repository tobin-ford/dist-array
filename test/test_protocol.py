import dist_array

from dist_array.core.protocol import (
    OP_CODE, 
    MSG_TYPE, 
    FLAGS, 
    DA_HEADER_FMT, 
    DA_HEADER_SIZE, 
    frame_message,
    parse_message
)

import pytest

def test_frame_and_parse_roundtrip():
    payload = b"\xde\xad\xbe\xef"
    msg = frame_message(
        OP_CODE.MAT_MUL,
        MSG_TYPE.REQUEST,
        FLAGS.TARGET_CPU | FLAGS.TARGET_GPU, # bitmask both
        seq_id=123456789,
        payload=payload,
    )
    assert len(msg) == DA_HEADER_SIZE + len(payload)

    total_len, op, mtype, flags, seq, pl_view = parse_message(msg)
    assert total_len == len(msg)
    assert mtype is MSG_TYPE.REQUEST
    assert op is OP_CODE.MAT_MUL
    assert seq == 123456789
    assert flags & FLAGS.TARGET_CPU
    assert flags & FLAGS.TARGET_GPU
    assert bytes(pl_view) == payload

def test_truncated_header_raises():
    with pytest.raises(ValueError):
        parse_message(b"\x00" * (DA_HEADER_SIZE - 1))

def test_truncated_payload_raises():
    # Build a message then chop off the last byte
    payload = b"\x00\x01\x02"
    msg = frame_message(OP_CODE.MAT_MUL, MSG_TYPE.RESPONSE, FLAGS.NONE, 7, payload)
    with pytest.raises(ValueError):
        parse_message(msg[:-1])

def test_enum_values_are_ints_no_commas():
    # sanity: ensure no trailing commas created tuples
    assert isinstance(OP_CODE.MAT_MUL.value, int)
    assert isinstance(MSG_TYPE.REQUEST.value, int)

def test_seq_id_wraps_to_32bit():
    payload = b""
    msg = frame_message(OP_CODE.MAT_MUL, MSG_TYPE.REQUEST, FLAGS.NONE, 0x1_0000_0001, payload)
    _, _, _, _, seq, _ = parse_message(msg)
    assert seq == 1  # wrapped modulo 2^32
