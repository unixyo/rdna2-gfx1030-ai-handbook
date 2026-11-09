# MOSS RDNA2 v0.3

Default launcher modes: INT8_LAYOUT=view, ATTENTION=auto. Source-based eager integration, FP32 input/output only; FP16/BF16 use the original eager entry.

Operational guide, benchmarks and rollback backups:
~/DevSpace/rdna2-pro-research-20261102/rdna2-ops/production/README.md

Rollback after stopping this MOSS server:

```bash
cd ~/DevSpace/moss-image-rdna2-lab01
bash scripts/stop_moss_comfy.sh
MOSS_RDNA2_INT8_PATCH=1 MOSS_RDNA2_INT8_LAYOUT=legacy MOSS_RDNA2_ATTENTION=native bash scripts/start_moss_comfy.sh
```

Keep PATCH=1 for the original gfx1030 INT8 GEMM workaround. No package or model rebuild is required.
