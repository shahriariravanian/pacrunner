"""pacrunner.uir: UIR v0.8 in Python (pac-julia's types.jl, uir.jl, validate.jl, recurrence_export.jl,
passes.jl and encode.jl). Indices are 0-based, an absent one None.

    from pacrunner import uir
    ls = uir.load_recurrence_export("export.json")   # one Lowered per program entry
    e = uir.encode(uir.optimize(ls[0]))              # the 8-byte words and the edge table
"""

from .encode import EncodeError, Encoded, decode, decode_edge, decode_rows, decode_stream, encode, encode_edge, fields
from .export import ExportError, export_kernels, load_recurrence_export, read_export, recurrence_references
from .passes import fuse, optimize, sort_runs
from .types import (TYPE_COMPONENTS, UIR_VERSION, Form, InputKind, Kernel, KernelInput, LCurrent, LEdge, LOp, Lowered, LStub,
                    Opcode, PhysicalType, is_kernel, is_propagator, is_vertex, no_destination, operand_edges, source_fields,
                    source_op, source_template)
from .validate import InvalidLowered, problems, validate

__all__ = [
    "EncodeError", "Encoded", "decode", "decode_edge", "decode_rows", "decode_stream", "encode", "encode_edge", "fields",
    "ExportError", "export_kernels", "load_recurrence_export", "read_export", "recurrence_references",
    "fuse", "optimize", "sort_runs",
    "TYPE_COMPONENTS", "UIR_VERSION", "Form", "InputKind", "Kernel", "KernelInput", "LCurrent", "LEdge", "LOp", "Lowered",
    "LStub", "Opcode", "PhysicalType", "is_kernel", "is_propagator", "is_vertex", "no_destination", "operand_edges",
    "source_fields", "source_op", "source_template",
    "InvalidLowered", "problems", "validate",
]
