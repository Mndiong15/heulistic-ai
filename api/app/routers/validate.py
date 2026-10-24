"""Config validation, shared with the CLI.

Both frontends call this instead of carrying their own copy of the Axolotl
rules — the same `heulistic.axolotl` module backs `heulistic validate-config`
and job submission, so all three agree on what a valid config is.

A broken config is a 200 with diagnostics, not a 4xx: the caller's job is to
show the user what is wrong, which a rejected request cannot do.
"""

from fastapi import APIRouter, Depends, HTTPException

from heulistic.axolotl import validate as validate_config

from ..dependencies import get_current_user
from ..models.db import User
from ..schemas.jobs import DiagnosticResponse, ValidateRequest, ValidateResponse
from ..services.cost import hardware_for
from ..services.instances import valid_instance_types

router = APIRouter(tags=["validate"])

_LEVELS = ("error", "warning", "info")


@router.post("/validate", response_model=ValidateResponse)
async def validate_config_endpoint(
    req: ValidateRequest,
    user: User = Depends(get_current_user),
):
    target = None
    if req.instance_type:
        if req.instance_type not in valid_instance_types():
            raise HTTPException(
                status_code=422,
                detail=f"Invalid instance_type '{req.instance_type}'. "
                       f"Valid options: {sorted(valid_instance_types())}",
            )
        target = hardware_for(req.instance_type)

    diagnostics = validate_config(req.config_yaml, target=target)
    counts = {
        level: sum(1 for d in diagnostics if d.level == level) for level in _LEVELS
    }

    return ValidateResponse(
        ok=counts["error"] == 0,
        counts=counts,
        diagnostics=[
            DiagnosticResponse(
                level=d.level,
                code=d.code,
                message=d.message,
                field=d.field,
                line=d.line,
            )
            for d in diagnostics
        ],
    )
