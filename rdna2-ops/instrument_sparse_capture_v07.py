"""Observe selected native Qwen calls without changing attention or its output."""
from __future__ import annotations
import functools
import hashlib
import importlib
import inspect
import json
from pathlib import Path
import torch

from sparse_switches_v07 import _geometry


def install(*, nodes, spec, state, out, run_stage, digest, save):
    if spec["model"] != "qwen":
        raise ValueError("The native observer is Qwen-specific")
    module=importlib.import_module("comfy.ldm.qwen_image21.model")
    original=module.optimized_attention
    cls=module.QwenImage21Transformer2DModel
    build_original=cls.build_sequence
    selected=spec["sparse_config"]["capture_selected_qkv"]
    dest=Path(out)/"sparse_capture";dest.mkdir(exist_ok=True)
    stats={"kind":"native_qkv_observer","attention_unchanged":True,"captures":[],
           "source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           "selection":selected,"timing_eligible":False}
    geometry=None;step=-1;request_index=None

    @functools.wraps(build_original)
    def build(self,x,context,ref_latents,image_slots):
        nonlocal geometry
        result=build_original(self,x,context,ref_latents,image_slots)
        geometry=_geometry(x,ref_latents,result[2])
        return result

    @functools.wraps(original)
    def attention(q,k,v,*args,**kw):
        nonlocal step,request_index
        if request_index != state.get("request_index"):
            step=-1;request_index=state.get("request_index")
        heads=args[0] if args and type(args[0]) is int else kw.get("heads")
        opts=kw.get("transformer_options",{})
        layer=opts.get("block_index") if isinstance(opts,dict) else None
        mask=kw.get("mask",kw.get("attn_mask"))
        observed=(all(isinstance(t,torch.Tensor) and t.ndim==3 for t in (q,k,v))
                  and geometry is not None and heads and mask is None
                  and q.shape[1]==geometry["target"]["end"]-geometry["target"]["start"]
                  and k.shape[1]==geometry["key_length"])
        if observed and layer==0:step+=1
        if observed and [step,layer] in selected:
            path=dest/f"representative_qkv_step{step:02d}_layer{layer:02d}.pt"
            torch.save({"q":q.detach().cpu(),"k":k.detach().cpu(),"v":v.detach().cpu(),
                        "mask":None,"heads":heads,"scale":kw.get("scale"),"geometry":geometry,
                        "metadata":{"actual_capture":True,"native_output_unchanged":True,
                            "step":step,"layer":layer,"shapes":[list(t.shape) for t in (q,k,v)],
                            "strides":[list(t.stride()) for t in (q,k,v)],"dtype":str(q.dtype),
                            "attention_source":inspect.getsourcefile(inspect.unwrap(original))}},path)
            stats["captures"].append({"path":str(path),"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
                                      "shape":[q.shape[1],k.shape[1]],"step":step,"layer":layer})
            state["candidate"]=stats
            (Path(out)/"qkv_capture.json").write_text(json.dumps(stats,indent=2)+"\n")
        return original(q,k,v,*args,**kw)

    cls.build_sequence=build
    module.optimized_attention=attention

    def cleanup():
        cls.build_sequence=build_original
        module.optimized_attention=original
        stats["last_step"]=step
        (Path(out)/"qkv_capture.json").write_text(json.dumps(stats,indent=2)+"\n")
    return cleanup
