from decimal import Decimal

from fastapi import APIRouter

from ..services.cost import MARKUP
from ..services.instances import get_catalog

router = APIRouter(prefix="/instances", tags=["instances"])


@router.get("")
async def list_instances():
    catalog = get_catalog()
    return [
        {
            "instance_type": itype,
            "gpus": info["gpus"],
            "vram_gb": info["vram_gb"],
            "our_rate": float(
                (Decimal(str(info["on_demand_rate"])) * MARKUP).quantize(Decimal("0.001"))
            ),
        }
        for itype, info in sorted(catalog.items(), key=lambda x: x[1]["on_demand_rate"])
    ]
