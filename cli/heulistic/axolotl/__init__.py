"""Axolotl config schema and validation, shared by the Heulistic CLI and API.

    from heulistic.axolotl import validate, Hardware

    for finding in validate(open("config.yaml").read()):
        print(finding.level, finding.code, finding.message)

Pass a `Hardware` target to also check the config fits the GPU it will run on.
"""

from .schema import SCHEMA, get_field, get_model, is_known_field, known_field_names
from .spec import Field, Reference, SubSchema, TypeExpr, parse_type
from .inference import (
    GGUF_QUANTS,
    export_scratch_gb,
    gguf_size_gb,
    inference_memory_gb,
    recommend_quant,
)
from .validate import Diagnostic, validate
from .vram import (
    Hardware,
    ParallelismPlan,
    parse_param_billions,
    plan_parallelism,
    training_bits,
    vram_required_gb,
)

__version__ = "0.1.1"

__all__ = [
    # Validation — the main entry point
    "validate",
    "Diagnostic",
    "Hardware",
    # Schema access
    "SCHEMA",
    "get_field",
    "get_model",
    "is_known_field",
    "known_field_names",
    # VRAM estimation and parallelism
    "vram_required_gb",
    "plan_parallelism",
    "ParallelismPlan",
    "parse_param_billions",
    "training_bits",
    # Running an exported model
    "GGUF_QUANTS",
    "export_scratch_gb",
    "gguf_size_gb",
    "inference_memory_gb",
    "recommend_quant",
    # Schema types
    "Field",
    "Reference",
    "SubSchema",
    "TypeExpr",
    "parse_type",
    "__version__",
]
