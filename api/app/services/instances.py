"""
GPU instance catalog: fetches specs from ec2:DescribeInstanceTypes and
on-demand prices from the AWS Pricing API at first call, then caches for the
process lifetime.

Falls back to a hardcoded table when LOCAL_MODE=true or AWS calls fail.
"""
import json
import logging
from functools import lru_cache

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from ..config import settings

logger = logging.getLogger(__name__)

# Instance types we can actually offer given current EC2 service quotas:
#   G/VT quota: 64 vCPUs  → max 64 vCPUs per instance (24xlarge/48xlarge exceed this)
#   P quota:     8 vCPUs  → no P-family instances can run
# Update this set after requesting and receiving quota increases.
_OFFERED: frozenset[str] = frozenset({
    # g4dn — T4 16 GB (4–64 vCPUs)
    "g4dn.xlarge", "g4dn.2xlarge", "g4dn.4xlarge",
    "g4dn.8xlarge", "g4dn.16xlarge", "g4dn.12xlarge",
    # g5 — A10G 22 GB (4–64 vCPUs)
    "g5.xlarge", "g5.2xlarge", "g5.4xlarge",
    "g5.8xlarge", "g5.16xlarge", "g5.12xlarge",
    # g6 — L4 22 GB (4–64 vCPUs)
    "g6.xlarge", "g6.2xlarge", "g6.4xlarge",
    "g6.8xlarge", "g6.16xlarge", "g6.12xlarge",
    # g6e — L40S 44 GB (4–64 vCPUs)
    "g6e.xlarge", "g6e.2xlarge", "g6e.4xlarge",
    "g6e.8xlarge", "g6e.16xlarge", "g6e.12xlarge",
})

_FALLBACK: dict[str, dict] = {
    # g4dn — T4 16 GB
    "g4dn.xlarge":   {"gpus": 1, "vram_gb": 16, "on_demand_rate": 0.526},
    "g4dn.2xlarge":  {"gpus": 1, "vram_gb": 16, "on_demand_rate": 0.752},
    "g4dn.4xlarge":  {"gpus": 1, "vram_gb": 16, "on_demand_rate": 1.204},
    "g4dn.8xlarge":  {"gpus": 1, "vram_gb": 16, "on_demand_rate": 2.264},
    "g4dn.16xlarge": {"gpus": 1, "vram_gb": 16, "on_demand_rate": 4.528},
    "g4dn.12xlarge": {"gpus": 4, "vram_gb": 64, "on_demand_rate": 3.912},
    # g5 — A10G 22 GB
    "g5.xlarge":     {"gpus": 1, "vram_gb": 22, "on_demand_rate": 1.006},
    "g5.2xlarge":    {"gpus": 1, "vram_gb": 22, "on_demand_rate": 1.212},
    "g5.4xlarge":    {"gpus": 1, "vram_gb": 22, "on_demand_rate": 1.624},
    "g5.8xlarge":    {"gpus": 1, "vram_gb": 22, "on_demand_rate": 2.248},
    "g5.16xlarge":   {"gpus": 1, "vram_gb": 22, "on_demand_rate": 4.096},
    "g5.12xlarge":   {"gpus": 4, "vram_gb": 88, "on_demand_rate": 5.672},
    # g6 — L4 22 GB (cheaper than g5)
    "g6.xlarge":     {"gpus": 1, "vram_gb": 22, "on_demand_rate": 0.805},
    "g6.2xlarge":    {"gpus": 1, "vram_gb": 22, "on_demand_rate": 0.978},
    "g6.4xlarge":    {"gpus": 1, "vram_gb": 22, "on_demand_rate": 1.323},
    "g6.8xlarge":    {"gpus": 1, "vram_gb": 22, "on_demand_rate": 2.013},
    "g6.16xlarge":   {"gpus": 1, "vram_gb": 22, "on_demand_rate": 3.394},
    "g6.12xlarge":   {"gpus": 4, "vram_gb": 88, "on_demand_rate": 4.602},
    # g6e — L40S 44 GB
    "g6e.xlarge":    {"gpus": 1, "vram_gb": 44,  "on_demand_rate": 1.861},
    "g6e.2xlarge":   {"gpus": 1, "vram_gb": 44,  "on_demand_rate": 2.373},
    "g6e.4xlarge":   {"gpus": 1, "vram_gb": 44,  "on_demand_rate": 3.397},
    "g6e.8xlarge":   {"gpus": 1, "vram_gb": 44,  "on_demand_rate": 5.445},
    "g6e.16xlarge":  {"gpus": 1, "vram_gb": 44,  "on_demand_rate": 9.541},
    "g6e.12xlarge":  {"gpus": 4, "vram_gb": 176, "on_demand_rate": 13.556},
}

_GPU_FAMILIES = ["g4dn", "g5", "g6", "g6e"]
_GPU_SIZES = ["xlarge", "2xlarge", "4xlarge", "8xlarge", "12xlarge", "16xlarge"]


def _fetch_gpu_specs(region: str) -> dict[str, dict]:
    ec2 = boto3.client("ec2", region_name=region)
    instance_types = [f"{f}.{s}" for f in _GPU_FAMILIES for s in _GPU_SIZES]
    paginator = ec2.get_paginator("describe_instance_types")
    specs: dict[str, dict] = {}
    for page in paginator.paginate(
        Filters=[{"Name": "instance-type", "Values": instance_types}]
    ):
        for it in page["InstanceTypes"]:
            gpu_list = it.get("GpuInfo", {}).get("Gpus", [])
            if not gpu_list:
                continue
            gpu = gpu_list[0]
            count = gpu.get("Count", 1)
            # vram_gb is the instance total, as in _FALLBACK; SizeInMiB is one card.
            total_mib = it["GpuInfo"].get("TotalGpuMemoryInMiB") or (
                gpu.get("MemoryInfo", {}).get("SizeInMiB", 0) * count
            )
            specs[it["InstanceType"]] = {
                "gpus": count,
                "vram_gb": total_mib // 1024,
            }
    return specs


def _fetch_on_demand_prices(region: str) -> dict[str, float]:
    pricing = boto3.client("pricing", region_name="us-east-1")
    paginator = pricing.get_paginator("get_products")
    rates: dict[str, float] = {}
    for page in paginator.paginate(
        ServiceCode="AmazonEC2",
        Filters=[
            {"Type": "TERM_MATCH", "Field": "operatingSystem", "Value": "Linux"},
            {"Type": "TERM_MATCH", "Field": "preInstalledSw",  "Value": "NA"},
            {"Type": "TERM_MATCH", "Field": "tenancy",         "Value": "Shared"},
            {"Type": "TERM_MATCH", "Field": "capacitystatus",  "Value": "Used"},
            {"Type": "TERM_MATCH", "Field": "regionCode",      "Value": region},
        ],
    ):
        for raw in page["PriceList"]:
            item = json.loads(raw)
            itype = item["product"]["attributes"].get("instanceType", "")
            if not itype:
                continue
            for term in item.get("terms", {}).get("OnDemand", {}).values():
                for dim in term.get("priceDimensions", {}).values():
                    price = float(dim.get("pricePerUnit", {}).get("USD", 0))
                    if price > 0:
                        rates[itype] = price
    return rates


@lru_cache(maxsize=1)
def get_catalog() -> dict[str, dict]:
    """
    Returns {instance_type: {gpus, vram_gb, on_demand_rate}} for offered instance types.
    Cached for the process lifetime. Falls back to hardcoded table on any failure.
    """
    if settings.local_mode:
        logger.info("LOCAL_MODE: using fallback instance catalog (%d types)", len(_FALLBACK))
        return dict(_FALLBACK)

    region = settings.aws_region
    try:
        specs = _fetch_gpu_specs(region)
        prices = _fetch_on_demand_prices(region)
        catalog = {
            itype: {**hw, "on_demand_rate": prices[itype]}
            for itype, hw in specs.items()
            if itype in prices and itype in _OFFERED
        }
        logger.info("Instance catalog loaded from AWS: %d types", len(catalog))
        return catalog
    except (BotoCoreError, ClientError, Exception) as exc:
        logger.warning("Instance catalog AWS fetch failed (%s) — using fallback", exc)
        return dict(_FALLBACK)


def get_rate(instance_type: str) -> float:
    entry = get_catalog().get(instance_type)
    if entry is None:
        raise KeyError(f"Unknown instance type: {instance_type!r}")
    return entry["on_demand_rate"]


def valid_instance_types() -> frozenset[str]:
    return frozenset(get_catalog())
