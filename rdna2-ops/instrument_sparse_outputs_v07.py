"""Archive actual candidate float tensors after the harness's VAE stage clock."""
from __future__ import annotations
import functools
import hashlib
from pathlib import Path
import torch


def install(*,nodes,spec,state,out,run_stage,digest,save):
    cls=getattr(nodes,spec["decoder"][0]);name=spec["decoder"][1]
    original=getattr(cls,name)
    @functools.wraps(original)
    def decode(self,*args,**kwargs):
        result=original(self,*args,**kwargs)
        if state.get("mode")=="sparse_radial":
            path=Path(out)/f"candidate_{state['request_index']:03d}_tensors.pt"
            torch.save({"latent":state["latent"],"image":state["image"],
                        "mode":state["mode"],"request_index":state["request_index"]},path)
            state["candidate"]["output_tensor_file"]={"path":str(path),
                "sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
                "artifact_write_outside_stage_clocks":True,
                "request_wall_includes_artifact_write":True}
        return result
    setattr(cls,name,decode)
    return lambda:setattr(cls,name,original)
