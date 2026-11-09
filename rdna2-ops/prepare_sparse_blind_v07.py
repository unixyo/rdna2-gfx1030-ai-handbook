#!/usr/bin/env python3
"""Create lossless, anonymous 1:1 A/B and B/A panels plus a public common task.

Input has pairs [{seed,reference,candidate}] and public_task
{prompt,negative_prompt,common_input}. Only the coordinator receives the private
mapping. Every output PNG is <=2,000,000 bytes with no original PNG metadata.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
from pathlib import Path
import random
from PIL import Image, ImageDraw

MAX_BYTES=2_000_000  # CodexPro2 view_image schema uses a decimal 2 MB cap.
CROPS=[("C1",(.05,.02,.95,.42)),("C2",(.05,.30,.95,.75)),("C3",(.05,.58,.95,.98))]
PROMPT="""먼저 PUBLIC_TASK.json의 공통 생성 지시와 공통 입력을 읽어 줘. 공통 입력은 생성 결과가 아니야. 두 익명 생성 이미지 A/B와 같은 좌표의 crop 세 쌍을 평가해 줘. 어느 쪽이 기준/후보인지 알려져 있지 않아. 속도나 수치 오차를 추측하지 말고 보이는 것만 판단해. 얼굴/손과 해부, 텍스트/작은 물체, 경계/윤곽, 질감/세부, 색/명암, 전체 구성 여섯 항목에서 A와 B의 결함 정도를 0(없음), 1(작음), 2(뚜렷함), 3(심함)으로 표시해. 공통 지시/입력에 대한 충실도도 별도로 설명해. 두 결과가 함께 갖는 결함과 어느 한쪽에만 보이는 열화를 구분해. 위치나 표정이 달라졌다는 이유만으로 화질 저하로 단정하지 않아도 돼. 마지막에는 A 우세/B 우세/지각적으로 동등/판정 불가 중 하나와 확신 정도, 구체적인 위치와 관찰 근거를 써 줘. 모든 full/crop 파일을 확인해. 크기 제한으로 나눈 파일은 source_region을 따라 같은 이미지의 부분으로 읽어. crop은 같은 좌표의 1:1 원본 픽셀이며 리사이즈가 없다. 모호한 부분은 모호하다고 밝혀 줘. 판정 결과는 독립 연구 기록으로 원문 그대로 보존돼."""


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def png_bytes(image):
    stream=io.BytesIO()
    image.save(stream,format="PNG",optimize=True,compress_level=9)
    return stream.getvalue()


class Assets:
    def __init__(self,out):
        self.out=out;self.index=0

    def save(self,canvas,regions,kind,box):
        blob=png_bytes(canvas)
        if len(blob)>MAX_BYTES:return None
        path=self.out/f"a{self.index:04d}.png";self.index+=1
        path.write_bytes(blob)
        # A roundtrip check proves source image pixels were neither resized
        # nor changed by compositing or PNG encoding.
        restored=Image.open(io.BytesIO(blob)).convert("RGB")
        for region,original in regions:
            if restored.crop(region).tobytes()!=original.tobytes():
                raise AssertionError("Panel changed source RGB pixels")
        return {"path":str(path),"sha256":sha(path),"bytes":len(blob),
                "view":kind,"source_region":list(box),"size":list(canvas.size),
                "pixel_identity_verified":True,"resized":False}

    def pair(self,a,b,kind,box):
        if a.size!=b.size:raise ValueError("Pair dimensions differ")
        w,h=a.size
        canvas=Image.new("RGB",(2*w+24,h+48),(245,245,245))
        draw=ImageDraw.Draw(canvas)
        draw.text((12,12),"A | "+kind,fill=(0,0,0))
        draw.text((w+36,12),"B | "+kind,fill=(0,0,0))
        canvas.paste(a,(0,40));canvas.paste(b,(w+24,40))
        item=self.save(canvas,[((0,40,w,40+h),a),((w+24,40,2*w+24,40+h),b)],kind,box)
        if item is not None:return [item]
        if kind=="full":
            # Keep each complete composition visible if a combined full pair
            # is too large. A and B remain 1:1, separately labelled, anonymous.
            return self.solo(a,"A",kind,box)+self.solo(b,"B",kind,box)
        cuts=self._cuts(w,h)
        result=[]
        for rect in cuts:
            absolute=(box[0]+rect[0],box[1]+rect[1],box[0]+rect[2],box[1]+rect[3])
            result.extend(self.pair(a.crop(rect),b.crop(rect),kind,absolute))
        return result

    def solo(self,image,side,kind,box):
        w,h=image.size
        canvas=Image.new("RGB",(w,h+48),(245,245,245))
        ImageDraw.Draw(canvas).text((12,12),side+" | "+kind,fill=(0,0,0))
        canvas.paste(image,(0,40))
        item=self.save(canvas,[((0,40,w,40+h),image)],kind,box)
        if item is not None:
            item["side"]=side
            item["pair_layout"]="separate_full_views_due_to_file_limit"
            return [item]
        result=[]
        for rect in self._cuts(w,h):
            absolute=(box[0]+rect[0],box[1]+rect[1],box[0]+rect[2],box[1]+rect[3])
            result.extend(self.solo(image.crop(rect),side,kind,absolute))
        return result

    def common(self,image,box):
        w,h=image.size
        canvas=Image.new("RGB",(w,h+48),(245,245,245))
        ImageDraw.Draw(canvas).text((12,12),"COMMON INPUT (not generated output)",fill=(0,0,0))
        canvas.paste(image,(0,40))
        item=self.save(canvas,[((0,40,w,40+h),image)],"common_input",box)
        if item is not None:return [item]
        result=[]
        for rect in self._cuts(w,h):
            absolute=(box[0]+rect[0],box[1]+rect[1],box[0]+rect[2],box[1]+rect[3])
            result.extend(self.common(image.crop(rect),absolute))
        return result

    @staticmethod
    def _cuts(w,h):
        if h>1:return [(0,0,w,h//2),(0,h//2,w,h)]
        if w>1:return [(0,0,w//2,h),(w//2,0,w,h)]
        raise ValueError("Single pixel PNG exceeded the asset limit")


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--random-seed",type=int,default=740711)
    args=ap.parse_args()
    if args.out.exists():raise FileExistsError("Blind output already exists")
    manifest=json.loads(args.pairs.read_text());pairs=manifest["pairs"];task=manifest["public_task"]
    if not task.get("prompt") or not task.get("common_input"):raise ValueError("Actual public task/input required")
    args.out.mkdir(parents=True);assets=Assets(args.out)
    rng=random.Random(args.random_seed);order=[]
    for index,pair in enumerate(pairs):
        forward=rng.choice((True,False));order.extend([(index,forward),(index,not forward)])
    rng.shuffle(order);private=[];public=[]
    for index,forward in order:
        pair=pairs[index];paths=[Path(pair["reference"]),Path(pair["candidate"])]
        if not forward:paths=paths[::-1]
        images=[Image.open(path).convert("RGB") for path in paths]
        if images[0].size!=images[1].size:raise ValueError("Source sizes differ")
        ident=f"p{rng.getrandbits(64):016x}";w,h=images[0].size
        files=assets.pair(*images,"full",(0,0,w,h));boxes=[]
        for label,box in CROPS:
            pixels=tuple(round(x*(w if i%2==0 else h)) for i,x in enumerate(box));boxes.append(pixels)
            files.extend(assets.pair(*(image.crop(pixels) for image in images),label,pixels))
        item={"pair_id":ident,"assets":files,"full":[x["path"] for x in files if x["view"]=="full"],
              "crops":[x["path"] for x in files if x["view"]!="full"]}
        public.append(item)
        private.append({**item,"seed":pair["seed"],"pair_index":index,"left_is_reference":forward,
                        "sources":[str(path) for path in paths],"source_sha256":[sha(path) for path in paths],
                        "source_size":images[0].size,"crop_pixels":boxes})
    common=Image.open(task["common_input"]).convert("RGB")
    common_files=assets.common(common,(0,0,common.width,common.height))
    public_task={"prompt":task["prompt"],"negative_prompt":task.get("negative_prompt",""),
                 "common_input_label":"공통 입력: 생성 결과가 아님","common_input_assets":common_files}
    (args.out/"PUBLIC_TASK.json").write_text(json.dumps(public_task,indent=2,ensure_ascii=False)+"\n")
    (args.out/"PUBLIC_CASES.json").write_text(json.dumps({"cases":public,"prompt_file":"JUDGE_PROMPT_KO.txt",
        "public_task_file":"PUBLIC_TASK.json","max_asset_bytes":MAX_BYTES,"source_pixels_1_to_1":True},indent=2)+"\n")
    (args.out/"PRIVATE_MAPPING.json").write_text(json.dumps({"random_seed":args.random_seed,"cases":private,
        "common_input_source":task["common_input"],"common_input_source_sha256":sha(task["common_input"]),
        "manifest_path":str(args.pairs),"manifest_sha256":sha(args.pairs)},indent=2)+"\n")
    (args.out/"JUDGE_PROMPT_KO.txt").write_text(PROMPT+"\n")
    print(json.dumps({"cases":len(public),"seeds":len(pairs),"assets":assets.index,"out":str(args.out),
                      "max_asset_bytes":MAX_BYTES,"source_pixels_1_to_1":True}))


if __name__=="__main__":main()

