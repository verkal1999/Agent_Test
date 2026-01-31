from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Dict


@dataclass
class GlobalVar:
    name: str
    type: str
    init: Optional[str] = None
    address: Optional[str] = None


@dataclass
class GVL:
    name: str
    globals: List[GlobalVar]


@dataclass
class IOEntry:
    internal: Optional[str]
    external: Optional[str]
    internal_type: Optional[str] = None


@dataclass
class TempEntry:
    name: str
    type: Optional[str] = None


@dataclass
class SubcallParam:
    internal: Optional[str]
    external: Optional[str]


@dataclass
class Subcall:
    SubNetwork_Name: str
    instanceName: Optional[str]
    inputs: List[SubcallParam]
    outputs: List[SubcallParam]


@dataclass
class ProgramMapping:
    programm_name: str
    inputs: List[IOEntry]
    outputs: List[IOEntry]
    inouts: List[IOEntry]
    temps: List[TempEntry]
    subcalls: List[Subcall]
    program_code: str = ""
    programming_lang: Optional[str] = None


@dataclass
class IoVarSide:
    raw: str
    parts: List[str]
    is_plc_task: bool
    channel: str


@dataclass
class IoLink:
    ownerA: str
    ownerB: str
    varA: str
    varB: str
    sideA: IoVarSide
    sideB: IoVarSide
    plc: Optional[IoVarSide]
    io: Optional[IoVarSide]


@dataclass
class IoHardwareAddress:
    plc_path: str
    plc_var: str
    device_path: str
    channel_label: str
    io_path: str
    direction: str
    ea_address: Optional[str]
    varBitAddr: Optional[int]
    varBitSize: Optional[int]
    varInOut: Optional[int]
    byte_offset: Optional[int]
    bit_offset: Optional[int]
    amsPort: Optional[int]
    indexGroup: Optional[int]
    indexGroupHex: Optional[str]
    indexOffset: Optional[int]
    indexOffsetHex: Optional[str]
    length: Optional[int]
    raw_offsets: Dict[str, int]
    io_raw_xml: str
