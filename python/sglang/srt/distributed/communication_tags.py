from enum import IntEnum, unique


@unique
class P2PTag(IntEnum):
    """Tags for point-to-point protocols outside the scheduler data path."""

    DEFAULT = 0
    HIRADIX_PP_SYNC = int.from_bytes(b"PpHi", byteorder="big")
    GRAMMAR_PP_SYNC = int.from_bytes(b"PpGr", byteorder="big")
