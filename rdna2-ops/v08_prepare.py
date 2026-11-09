#!/usr/bin/env python3
"""Create a bounded Applio v0.8 research copy; never alter operating sources/data."""
from pathlib import Path
import hashlib,json,shutil,os
ROOT=Path('~/DevSpace/applio-lab')
OPS=Path(__file__).resolve().parent
OUT=OPS/'results/applio_v08'
COPY=OUT/'applio_copy'
DATA='voice_model_A'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 OUT.mkdir(parents=True,exist_ok=True)
 if COPY.exists():raise RuntimeError('Existing research copy: refusing overwrite')
 legacy=[ROOT/'assets/config.json',ROOT/'logs/bench40k/config.json',ROOT/'logs/bench40k/filelist.txt',ROOT/'logs/bench40k/model_info.json',ROOT/'rvc/train/train.py',ROOT/'rvc/lib/zluda.py',ROOT/'rvc/lib/algorithm/fold_conv.py',ROOT/'lab/train_bench.sh',ROOT/'core.py',ROOT/'rvc/train/lab_hooks.py',ROOT/'rvc/train/data_utils.py',ROOT/'rvc/lib/algorithm/synthesizers.py',ROOT/'rvc/lib/algorithm/discriminators.py',ROOT/'rvc/lib/algorithm/residuals.py',ROOT/'rvc/models/pretraineds/hifi-gan/f0G40k.pth',ROOT/'rvc/models/pretraineds/hifi-gan/f0D40k.pth']
 extra=[ROOT/'run-applio.sh']+sorted(p for p in (ROOT/'logs'/DATA).rglob('*') if p.is_file())+[ROOT/'rvc/models/pretraineds/custom/G_KLM_HFG_40k.pth',ROOT/'rvc/models/pretraineds/custom/D_KLM_HFG_40k.pth']+sorted((ROOT/'rvc').rglob('*.py'))
 extra=sorted(set(extra)-set(legacy))
 manifest={'schema':'applio_v08_protected','legacy16':{str(p):sha(p) for p in legacy},'additional_target_and_source':{str(p):sha(p) for p in extra}}
 (OUT/'protected_before.json').write_text(json.dumps(manifest,indent=2))
 copied=[]
 def cp(src,dst):
  dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
  h=sha(src)
  if h!=sha(dst):raise RuntimeError('Copy mismatch '+str(src))
  copied.append({'source':str(src),'copy':str(dst),'bytes':src.stat().st_size,'sha256':h})
 for p in (ROOT/'rvc').rglob('*'):
  if p.is_file() and p.suffix in {'.py','.json','.yaml','.yml','.txt'} and '__pycache__' not in p.parts:
   cp(p,COPY/p.relative_to(ROOT))
 for rel in ['assets/config.json','assets/config_template.json','core.py']:
  if (ROOT/rel).exists():cp(ROOT/rel,COPY/rel)
 for name in ['config.json','filelist.txt','model_info.json']:
  cp(ROOT/'logs'/DATA/name,COPY/'logs'/DATA/name)
 lines=(ROOT/'logs'/DATA/'filelist.txt').read_text().splitlines()
 files=sorted({ROOT/part for line in lines for part in line.split('|')[:4]})
 for p in files:
  if not p.resolve().is_relative_to(ROOT.resolve()):raise RuntimeError('Unexpected data path '+str(p))
  cp(p,COPY/p.relative_to(ROOT))
 for name in ['G_KLM_HFG_40k.pth','D_KLM_HFG_40k.pth']:
  cp(ROOT/'rvc/models/pretraineds/custom'/name,COPY/'rvc/models/pretraineds/custom'/name)
 (OUT/'copy_manifest.json').write_text(json.dumps(copied,indent=2))
 gates={'operator':{'finite':['y','dX','dW'],'nrmse_max':0.002,'max_abs_div_reference_rms_max':0.02,'reference':'FP32 same fixedseed stored input'},
 'epoch':{'same_session_alternating':True,'metrics':['steady_epoch_median','first_epoch','scale_decreases','nonfinite_losses','loss_median_relative_delta'],'quality_not_proven_by_short_epoch':True},
 'final':{'runs':['A current','A_prime current repeat','B candidate'],'epochs_each':100,'batch':8,'dataset':DATA,'pretrain':'KLM HFG 40k','perceptual':'anonymous A/B/C same input sentences; user rejects B-only artifacts beyond A/A-prime variation','objective_reference_only':['last25percent mel/fm/kl/adv medians','audio saturation']},
 'safety_env':{'MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_FWD_V4R1':'0','MIOPEN_DEBUG_CONV_DIRECT_HIP_WRW53':'0'},
 'dataset_filelist_lines':len(lines),'dataset_unique_wavs':len({x.split('|')[0] for x in lines}),'operating_config':json.loads((ROOT/'assets/config.json').read_text())}
 (OUT/'PREDECLARED_GATES.json').write_text(json.dumps(gates,indent=2))
 print(json.dumps({'copy':str(COPY),'files':len(copied),'bytes':sum(x['bytes'] for x in copied),'legacy16':len(manifest['legacy16']),'additional':len(extra),'filelist_lines':len(lines),'gates':str(OUT/'PREDECLARED_GATES.json')},indent=2))
if __name__=='__main__':main()
