"""TunableOp on/off switch for the owned research harness (Claude, 2026-11-07).

Mode "t_<inner>" runs <inner> with PyTorch TunableOp enabled (tuning on, so new
shapes are tuned on first use). "t_baseline" is the plain request with TunableOp.
Other modes are delegated to int4_model_switches_v07 unchanged.
"""
from contextlib import contextmanager
import torch
import int4_model_switches_v07 as inner_switches


@contextmanager
def activate(mode, spec, state, bound, out):
    tunable = mode.startswith("t_")
    inner = mode[2:] if tunable else mode
    torch.cuda.tunable.enable(tunable)
    torch.cuda.tunable.tuning_enable(tunable)
    try:
        if inner == "baseline":
            state["candidate"] = {"mode": mode, "tunableop": tunable}
            yield
        else:
            with inner_switches.activate(inner, spec, state, bound, out):
                state.setdefault("candidate", {})["tunableop"] = tunable
                yield
    finally:
        torch.cuda.tunable.enable(False)
        torch.cuda.tunable.tuning_enable(False)
