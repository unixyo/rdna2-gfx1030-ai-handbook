#!/usr/bin/env python3
"""Full pretrained VAE decoder A/B; existing weights, fixed synthetic latents."""
import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import statistics
import sys

import torch
from safetensors.torch import load_file
from rdna2_ops.comfy import fuse_comfy_vae_decoder
from validate_bench import timed_call


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--comfy",type=Path,required=True)
    ap.add_argument("--checkpoint",type=Path,required=True)
    ap.add_argument("--output",type=Path,default=Path("results/pretrained_vae.json"))
    args=ap.parse_args()
    sys.path.insert(0,str(args.comfy))
    saved=sys.argv;sys.argv=["rdna2_pretrained_vae_benchmark"]
    from comfy import diffusers_convert
    from comfy.ldm.modules.diffusionmodules import model
    sys.argv=saved
    torch.set_num_threads(2)
    torch.manual_seed(20261105)
    digest=hashlib.sha256()
    with args.checkpoint.open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''):digest.update(chunk)
    sd=diffusers_convert.convert_vae_state_dict(load_file(str(args.checkpoint),device="cpu"))
    # This decoder layout is verified by strict state-dict loading below.
    config=dict(ch=96,out_ch=3,ch_mult=(1,2,4,4),num_res_blocks=2,attn_resolutions=[],dropout=0.0,in_channels=3,resolution=256,z_channels=32)
    decoder=model.Decoder(**config)
    decoder.load_state_dict({k[len('decoder.'):]:v for k,v in sd.items() if k.startswith('decoder.')},strict=True)
    post_quant=torch.nn.Conv2d(32,32,1)
    post_quant.load_state_dict({k[len('post_quant_conv.'):]:v for k,v in sd.items() if k.startswith('post_quant_conv.')},strict=True)
    del sd
    decoder.eval().to(device='cuda',dtype=torch.float32)
    post_quant.eval().to(device='cuda',dtype=torch.float32)
    original_module_ids={n:id(m) for n,m in decoder.named_modules()}
    report={"checkpoint":str(args.checkpoint),"checkpoint_bytes":args.checkpoint.stat().st_size,"checkpoint_sha256":digest.hexdigest(),"weight_loading":"strict, all decoder and post_quant tensors", "dtype":"float32", "latent_source":"seeded synthetic standard-normal latent, seed20261105; not captured denoiser output", "workload":"entire pretrained post_quant+decoder+RGB postprocess", "source":str(Path(model.__file__)),"source_sha256":hashlib.sha256(Path(model.__file__).read_bytes()).hexdigest(),"acceptance":{"pixel_max_abs":2e-4,"pixel_rmse":2e-5},"rows":[]}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    try:
        with torch.inference_mode():
            for side in (32,64):
                z=torch.randn(1,32,side,side,device='cuda')
                def decode():
                    return decoder(post_quant(z)).add(1).mul(.5).clamp(0,1)
                native=decode()
                with fuse_comfy_vae_decoder(decoder) as count:
                    fused=decode()
                torch.cuda.synchronize()
                diff=(fused-native).abs()
                maximum=diff.max().item();rmse=diff.square().mean().sqrt().item()
                correct=bool(torch.isfinite(native).all() and torch.isfinite(fused).all() and maximum<=2e-4 and rmse<=2e-5)
                row={"latent_shape":list(z.shape),"rgb_shape":list(native.shape),"fused_blocks":count,"pixel_max_abs":maximum,"pixel_rmse":rmse,"pixel_8bit_max_abs":maximum*255,"accuracy_pass":correct}
                if not correct:
                    report['rows'].append(row);raise AssertionError(f"Pretrained VAE output criterion failed: {row}")
                timings={"native":[],"hip":[]}
                for mode in timings:
                    with (fuse_comfy_vae_decoder(decoder) if mode=='hip' else nullcontext()):
                        for _ in range(2):decode()
                for i in range(5):
                    for mode in (('native','hip') if i%2==0 else ('hip','native')):
                        with (fuse_comfy_vae_decoder(decoder) if mode=='hip' else nullcontext()):
                            timings[mode].append(timed_call(decode,3))
                row['samples']=timings
                row['native_wall_ms']=statistics.median(t['wall_ms'] for t in timings['native'])
                row['hip_wall_ms']=statistics.median(t['wall_ms'] for t in timings['hip'])
                row['speedup']=row['native_wall_ms']/row['hip_wall_ms']
                report['rows'].append(row)
                print(json.dumps({k:v for k,v in row.items() if k!='samples'}),flush=True)
                args.output.write_text(json.dumps(report,indent=2)+'\n')
        report['modules_restored']=original_module_ids=={n:id(m) for n,m in decoder.named_modules()}
        report['source_unchanged']=hashlib.sha256(Path(model.__file__).read_bytes()).hexdigest()==report['source_sha256']
        report['status']='passed'
    finally:
        args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(str(args.output.resolve()),flush=True)


if __name__=='__main__':main()
