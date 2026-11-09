"""Owned-process Qwen sparse research adapter and step/layer sparsity capture."""
from collections import Counter
from contextlib import contextmanager
import functools
import hashlib
import importlib
import json
from pathlib import Path
import torch


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _qh(t, heads):
    return t.reshape(t.shape[0],t.shape[1],heads,-1).transpose(1,2)


def _geometry(x, refs, segments):
    image_shapes=[list(r.shape[-2:]) for r in list(refs or [])]+[list(x.shape[-2:])]
    image_segs=[];text_segs=[]
    for start,end,mask in segments:
        if mask is None:
            hh,ww=image_shapes[len(image_segs)]
            image_segs.append({"start":start,"end":end,"height":hh,"width":ww})
        else:
            text_segs.append({"start":start,"end":end})
    return {"image_segments":image_segs,"text_segments":text_segs,"target":image_segs[-1],
            "key_length":segments[-1][1],"source":"actual Qwen build_sequence output and x/ref_latents shapes"}


def _masks(q,k,geometry):
    from rdna2_ops.sparse_attention_v07 import sparge_map,radial_map,fixed_map
    result={}
    for mass in (.9,.95,.99):
        result[f"sparge_p{mass:g}_c0.1"]=sparge_map(q,k,mass=mass,similarity=.1)
    result["sparge_p0.95_c0.5"]=sparge_map(q,k,mass=.95,similarity=.5)
    result["fixed_50"]=fixed_map(q,k,fraction=.5)
    if geometry and q.shape[-2]==geometry["target"]["end"]-geometry["target"]["start"] and k.shape[-2]==geometry["key_length"]:
        for radius in (4.,8.,16.):
            result[f"radial_r{radius:g}"]=radial_map(q,k,geometry,radius=radius)
    return result


@contextmanager
def activate(mode,spec,state,bound,out):
    """No process-global native source edit; all hooks restored in finally."""
    if spec["model"]!="qwen":
        raise ValueError("The model adapter is validated for Qwen; the kernel API is explicit")
    from rdna2_ops.sparse_attention_v07 import sparse_attention,sparge_map,radial_map,fixed_map,dense_alpha,map_statistics
    module=importlib.import_module("comfy.ldm.qwen_image21.model")
    original=module.optimized_attention
    cls=module.QwenImage21Transformer2DModel
    build_original=cls.build_sequence
    stats={"mode":mode,"hits":0,"fallbacks":0,"prefix_dense_calls":0,"shapes":Counter(),"map_summaries":[],"captures":[],
           "source_sha256":{str(p.name):_sha(p) for p in (Path(__file__),Path(__file__).parent/"rdna2_ops/sparse_attention_v07.py")},
           "baseline":"FP32 attention_split alpha folding, head-contiguous Q/K/V; native masked calls preserved",
           "selector_in_stage_timing":True,"first_candidate_call_in_stage_timing":True}
    state["candidate"]=stats
    native_path=Path(inspect_path(original))
    stats["native_callable"]={"module":original.__module__,"qualname":original.__qualname__,
        "source":str(native_path),"source_sha256":_sha(native_path) if native_path.is_file() else None,
        "masked_calls":"original callable receives the original mask tensor unchanged"}
    cfg=spec.get("sparse_config",{})
    kernel_cfg={"block_q":cfg.get("block_q",32),"num_warps":cfg.get("num_warps",8)}
    stats["kernel_config"]=kernel_cfg
    mode_keys={"sparse_sparge":"sparge","sparse_radial":"radial","sparse_fixed":"fixed","sparse_dense":"dense"}
    capture_mode=mode=="sparse_capture"
    if mode not in (*mode_keys,"dense_alpha","sparse_capture"):
        raise ValueError(mode)
    geometry=None;static_map=None;step=-1;seen_shapes=set();density_samples=[]
    capture_dir=Path(out)/"sparse_capture"
    if capture_mode:
        capture_dir.mkdir(exist_ok=True)

    @functools.wraps(build_original)
    def build(self,x,context,ref_latents,image_slots):
        nonlocal geometry
        result=build_original(self,x,context,ref_latents,image_slots)
        geometry=_geometry(x,ref_latents,result[2])
        stats["geometry"]=geometry
        return result

    @functools.wraps(original)
    def invoke(q,k,v,*args,**kw):
        nonlocal step,static_map
        heads=args[0] if args and type(args[0]) is int else kw.get("heads")
        mask=kw.get("mask",kw.get("attn_mask"))
        opts=kw.get("transformer_options",{})
        supported=(len(args)<=1 and isinstance(opts,dict) and not opts.get("optimized_attention_override")
                   and all(isinstance(t,torch.Tensor) and t.ndim==3 and t.dtype==torch.float32 for t in (q,k,v))
                   and heads and q.shape[0]==k.shape[0]==v.shape[0] and k.shape==v.shape and q.shape[-1]==k.shape[-1]
                   and q.shape[-1]//heads in (64,128) and not kw.get("skip_reshape") and not kw.get("skip_output_reshape")
                   and not kw.get("dropout_p",0) and not kw.get("enable_gqa",False) and not kw.get("is_causal",False))
        if not supported:
            stats["fallbacks"]+=1
            return original(q,k,v,*args,**kw)
        qh,kh,vh=[_qh(t,heads) for t in (q,k,v)]
        shape=(q.shape[1],k.shape[1]);layer=opts.get("block_index")
        target=(geometry is not None and shape==(geometry["target"]["end"]-geometry["target"]["start"],geometry["key_length"]) and mask is None)
        if target and layer==0:
            step+=1
        if capture_mode and cfg.get("capture_first_shapes",True) and shape not in seen_shapes and min(shape)>128:
            seen_shapes.add(shape)
            path=capture_dir/f"actual_qkv_{shape[0]}_{shape[1]}.pt"
            torch.save({"q":q.detach().cpu(),"k":k.detach().cpu(),"v":v.detach().cpu(),
                        "mask":mask.detach().cpu() if isinstance(mask,torch.Tensor) else None,
                        "heads":heads,"scale":kw.get("scale"),"geometry":geometry,
                        "metadata":{"actual_capture":True,"shapes":[list(t.shape) for t in(q,k,v)],
                                    "strides":[list(t.stride()) for t in(q,k,v)],"dtype":str(q.dtype),"layer":layer,"step":step,
                                    "attention_source":str(Path(inspect_path(original))) }},path)
            stats["captures"].append({"path":str(path),"sha256":_sha(path),"shape":shape,"layer":layer,"step":step})
        if capture_mode and target and [step,layer] in cfg.get("capture_selected_qkv",[]):
            path=capture_dir/f"representative_qkv_step{step:02d}_layer{layer:02d}.pt"
            torch.save({"q":q.detach().cpu(),"k":k.detach().cpu(),"v":v.detach().cpu(),
                        "mask":None,"heads":heads,"scale":kw.get("scale"),"geometry":geometry,
                        "metadata":{"actual_capture":True,"shapes":[list(t.shape) for t in(q,k,v)],
                                    "strides":[list(t.stride()) for t in(q,k,v)],"dtype":str(q.dtype),
                                    "layer":layer,"step":step,"attention_source":str(Path(inspect_path(original))) }},path)
            stats["captures"].append({"path":str(path),"sha256":_sha(path),"shape":shape,"layer":layer,"step":step})
        if capture_mode and cfg.get("capture_maps",True) and target and step in cfg.get("capture_steps",[0,3,7]) and layer in cfg.get("capture_layers",[0,15,31]):
            masks=_masks(qh,kh,geometry)
            summary,block_mass=map_statistics(qh,kh,masks,scale=kw.get("scale"))
            _,details=sparge_map(qh,kh,mass=.95,similarity=.1,return_details=True)
            details={key:value.detach().cpu() for key,value in details.items()}
            summary["selective_compression"]={}
            for key in ("query_cosine","key_cosine"):
                values=details[key].double().flatten()
                summary["selective_compression"][key]={"mean":values.mean().item(),"min":values.min().item(),
                    "median":values.median().item(),"p99":torch.quantile(values,.99).item(),
                    "fraction_forced_dense_theta0.1":(values<.1).double().mean().item(),
                    "fraction_forced_dense_theta0.5":(values<.5).double().mean().item()}
            summary.update(step=step,layer=layer,q_shape=list(qh.shape),k_shape=list(kh.shape))
            path=capture_dir/f"block_mass_step{step:02d}_layer{layer:02d}.pt"
            torch.save({"block_mass":block_mass,"summary":summary,"geometry":geometry,"selector_details":details,
                        "masks":{name:bm.keep.detach().cpu() for name,bm in masks.items()}},path)
            summary.update(raw_path=str(path),raw_sha256=_sha(path))
            stats["map_summaries"].append(summary)
            (capture_dir/"map_summaries.json").write_text(json.dumps(stats["map_summaries"],indent=2)+"\n")
        if mask is not None:
            stats["fallbacks"]+=1
            return original(q,k,v,*args,**kw)
        # Only the observed target query is sparsified; prefix construction and
        # text masks retain their original semantics and cached data.
        if mode in ("dense_alpha","sparse_capture") or not target:
            result=dense_alpha(qh,kh,vh,scale=kw.get("scale"))
            if not target and mode not in ("dense_alpha","sparse_capture"):
                stats["prefix_dense_calls"]+=1
        else:
            strategy=mode_keys[mode]
            if strategy=="sparge":
                bm=sparge_map(qh,kh,mass=cfg.get("mass",.95),similarity=cfg.get("similarity",.1))
            elif strategy=="radial":
                if static_map is None:
                    static_map=radial_map(qh,kh,geometry,radius=cfg.get("radius",8.))
                bm=static_map
            elif strategy=="fixed":
                if static_map is None:
                    static_map=fixed_map(qh,kh,fraction=cfg.get("fraction",.5))
                bm=static_map
            else:
                from rdna2_ops.sparse_attention_v07 import BlockMap
                if static_map is None:
                    static_map=BlockMap(torch.ones((qh.shape[0],qh.shape[1],(qh.shape[-2]+127)//128,(kh.shape[-2]+63)//64),device=qh.device,dtype=torch.bool))
                bm=static_map
            # Keep GPU counters until scope exit; avoid per-call host sync.
            density_samples.append(bm.keep.float().mean())
            result=sparse_attention(qh,kh,vh,bm,scale=kw.get("scale"),**kernel_cfg)
        stats["hits"]+=1
        stats["shapes"][str((tuple(qh.shape),tuple(kh.shape)))]+=1
        return result.transpose(1,2).reshape(q.shape)

    cls.build_sequence=build
    module.optimized_attention=invoke
    try:
        yield
    finally:
        module.optimized_attention=original
        cls.build_sequence=build_original
        stats["shapes"]=dict(stats["shapes"])
        if density_samples:
            values=torch.stack(density_samples).detach().cpu()
            stats["sparse_calls"]=len(values)
            stats["block_density"]={"mean":values.mean().item(),"min":values.min().item(),"max":values.max().item()}
        (Path(out)/f"sparse_{state.get('request_index',0):03d}_{mode}.json").write_text(json.dumps(stats,indent=2)+"\n")


def inspect_path(fn):
    import inspect
    return inspect.getsourcefile(inspect.unwrap(fn)) or "unknown"
