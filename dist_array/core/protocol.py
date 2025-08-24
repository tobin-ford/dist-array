"""
Protocol Spec for dist_array's multiple target execution environments.

Below transport layer protocol.
"""


import struct
from enum import Enum, unique, IntFlag

ENDIAN = "!"
DA_HEADER_FMT = ENDIAN + "LBBHL"
DA_HEADER_SIZE = struct.calcsize(DA_HEADER_FMT)
# described in README.md
# total length (4 bytes), msg type (1 byte), flags (1 byte), opcode (2 bytes), seq_id (4 bytes)

@unique
class OP_CODE(Enum):
    MAT_MUL=0x0001

class FLAGS(IntFlag):
    NONE = 0
    TARGET_CPU=  1 << 0
    TARGET_GPU=  1 << 1
    TARGET_FPGA= 1 << 2

@unique
class MSG_TYPE(Enum):
    REQUEST=0x01
    RESPONSE=0x02
    ERROR=0x03


def frame_message(
    op: OP_CODE, 
    msg_type: MSG_TYPE, 
    flags: FLAGS, 
    seq_id: int, 
    payload: bytes | bytearray | memoryview | None 
) -> bytes:
    """
    Prepend a message with protocol specific header information as described in README.md.

    OP_CODE
    MSG_TYPE
    FLAGS
    SEQUENCE_ID
    """

    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise TypeError("Payload must be bytes-like")

    total_len = DA_HEADER_SIZE + len(payload)
    header_bytes = struct.pack(
        DA_HEADER_FMT, 
        total_len, 
        msg_type.value, 
        flags.value, 
        op.value, 
        seq_id & 0xFFFFFFFF # unbounded python ints, make sure it fits into 4 bytes
    )

    return header_bytes + payload

def parse_message(
    buf: bytes | bytearray | memoryview
) -> tuple[int, OP_CODE, MSG_TYPE, FLAGS, int, bytes | bytearray | memoryview]:
    """
    Parse header protocol information before transport.
    """

    mv = memoryview(buf)
    
    if mv.nbytes < DA_HEADER_SIZE:
        raise ValueError("truncated: missing header")

    total_len, recv_msg_type, recv_flags, recv_op, recv_seq_id = struct.unpack_from(DA_HEADER_FMT, mv, offset=0)

    if mv.nbytes < total_len:
        raise ValueError("truncated: buffer shorter than message length")

    payload = mv[DA_HEADER_SIZE:total_len]

    msg_type = MSG_TYPE(recv_msg_type)
    flags = FLAGS(recv_flags)
    op_code = OP_CODE(recv_op)

    return total_len, op_code, msg_type, flags, recv_seq_id, payload