#!/usr/bin/env python3
import cv2, json, argparse, os, re
import numpy as np
from pathlib import Path

def split_into_tiles(img, cols, rows, overlap):
    H,W=img.shape[:2]
    tile_w=(W+overlap*(cols-1))//cols; tile_h=(H+overlap*(rows-1))//rows
    tiles=[]
    for r in range(rows):
        for c in range(cols):
            x0=c*(tile_w-overlap); y0=r*(tile_h-overlap)
            x1=min(x0+tile_w,W); y1=min(y0+tile_h,H)
            x0=max(0,x1-tile_w) if x1==W else x0
            y0=max(0,y1-tile_h) if y1==H else y0
            tiles.append((img[y0:y1, x0:x1].copy(), x0,y0, f"r{r+1}c{c+1}_x{x0}_y{y0}"))
    return tiles

def has_dashdot_pattern(tile_gray, x1,y1,x2,y2, th_img):
    length=int(np.hypot(x2-x1,y2-y1))
    if length < 60: return False
    xs=np.linspace(x1,x2,100).astype(int); ys=np.linspace(y1,y2,100).astype(int)
    h,w=th_img.shape[:2]
    vals=[]
    for x,y in zip(xs,ys):
        if 0 <= y < h and 0 <= x < w:
            vals.append(th_img[y,x] > 0)
    if not vals: return False
    transitions=sum(1 for i in range(1,len(vals)) if vals[i]!=vals[i-1])
    return 4 <= transitions <= 14

def detect_all(tile, x0,y0):
    gray=cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
    clahe=cv2.createCLAHE(2.0,(8,8)); eq=clahe.apply(gray)
    th=cv2.adaptiveThreshold(eq,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,41,12)
    edges=cv2.Canny(cv2.morphologyEx(th,cv2.MORPH_OPEN, np.ones((3,3),np.uint8)),50,150)
    out=[]
    lsd=cv2.createLineSegmentDetector()
    lines,_,_,_=lsd.detect(gray)
    if lines is not None:
        for l in lines:
            x1,y1,x2,y2=l[0]; length=float(np.hypot(x2-x1,y2-y1))
            if length<35: continue
            ang=float(np.degrees(np.arctan2(y2-y1,x2-x1)))
            out.append((x1+x0,y1+y0,x2+x0,y2+y0,length,ang,"straight", float(x1),float(y1),float(x2),float(y2)))
    hlines=cv2.HoughLinesP(edges,1,np.pi/180,threshold=70,minLineLength=50,maxLineGap=4)
    if hlines is not None:
        for x1,y1,x2,y2 in [l[0] for l in hlines]:
            length=float(np.hypot(x2-x1,y2-y1)); ang=float(np.degrees(np.arctan2(y2-y1,x2-x1)))
            out.append((x1+x0,y1+y0,x2+x0,y2+y0,length,ang,"straight", float(x1),float(y1),float(x2),float(y2)))
    hsv=cv2.cvtColor(tile, cv2.COLOR_BGR2HSV)
    mask=cv2.inRange(hsv, np.array([90,25,60]), np.array([128,255,255]))
    mask=cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8), iterations=2)
    blines=cv2.HoughLinesP(mask,1,np.pi/180,threshold=40,minLineLength=30,maxLineGap=25)
    if blines is not None:
        for x1,y1,x2,y2 in [l[0] for l in blines]:
            length=float(np.hypot(x2-x1,y2-y1)); ang=float(np.degrees(np.arctan2(y2-y1,x2-x1)))
            out.append((x1+x0,y1+y0,x2+x0,y2+y0,length,ang,"blue_dashed", float(x1),float(y1),float(x2),float(y2)))
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
    mask_dil=cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=1)
    edges2=cv2.Canny(mask_dil,50,150)
    dlines=cv2.HoughLinesP(edges2,1,np.pi/180,threshold=80,minLineLength=80,maxLineGap=30)
    if dlines is not None:
        for x1,y1,x2,y2 in [l[0] for l in dlines]:
            length=float(np.hypot(x2-x1,y2-y1))
            if length<80: continue
            ang=float(np.degrees(np.arctan2(y2-y1,x2-x1)))
            if not (abs(ang) < 12 or 55 < abs(ang) < 80 or abs(abs(ang)-90) < 12):
                continue
            if has_dashdot_pattern(tile, x1,y1,x2,y2,th):
                out.append((x1+x0,y1+y0,x2+x0,y2+y0,length,ang,"dashdot", float(x1),float(y1),float(x2),float(y2)))
    contours,_=cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        if cv2.arcLength(cnt, False) < 80: continue
        approx=cv2.approxPolyDP(cnt, 2.0, False)
        if len(approx)<2: continue
        pts=approx.reshape(-1,2)
        for i in range(len(pts)-1):
            x1,y1=pts[i]; x2,y2=pts[i+1]
            length=float(np.hypot(x2-x1,y2-y1))
            if length<25: continue
            ang=float(np.degrees(np.arctan2(y2-y1,x2-x1)))
            out.append((float(x1+x0),float(y1+y0),float(x2+x0),float(y2+y0),length,ang,"curved", float(x1),float(y1),float(x2),float(y2)))
    return out

def main():
    ap=argparse.ArgumentParser(description="Детектор линий для геологического разреза - координаты от левого верхнего угла")
    ap.add_argument("image")
    ap.add_argument("--tiles", default="3x2")
    ap.add_argument("--overlap", type=int, default=250)
    ap.add_argument("--output", default="lines.json")
    ap.add_argument("--debug-dir", default="debug")
    args=ap.parse_args()
    img=cv2.imread(args.image); H,W=img.shape[:2]
    print(f"Исходник {W}x{H} origin top-left (0,0)")
    cols,rows=map(int,args.tiles.split("x"))
    tiles=split_into_tiles(img,cols,rows,args.overlap)
    os.makedirs(args.debug_dir, exist_ok=True)
    all_lines=[]; seen=set()
    for tile,x0,y0,tid in tiles:
        cand=detect_all(tile,x0,y0)
        filtered=[]
        for x1,y1,x2,y2,length,ang,typ,lx1,ly1,lx2,ly2 in cand:
            key=(round(x1/2),round(y1/2),round(x2/2),round(y2/2),typ)
            if key in seen: continue
            seen.add(key)
            filtered.append({"x1":int(x1),"y1":int(y1),"x2":int(x2),"y2":int(y2),"length":round(length,1),"angle":round(ang,1),"type":typ,"tile":tid})
        all_lines.extend(filtered)
        print(f"  {tid}: straight {len([x for x in filtered if x['type']=='straight'])} blue {len([x for x in filtered if x['type']=='blue_dashed'])} dashdot {len([x for x in filtered if x['type']=='dashdot'])} curved {len([x for x in filtered if x['type']=='curved'])}")
        dbg=tile.copy()
        for f in filtered:
            lx1,ly1=f["x1"]-x0, f["y1"]-y0; lx2,ly2=f["x2"]-x0, f["y2"]-y0
            col=(0,165,255) if f["type"]=="dashdot" else (0,255,0) if f["type"]=="curved" else (255,0,0) if f["type"]=="blue_dashed" else (0,0,255)
            cv2.line(dbg,(int(lx1),int(ly1)),(int(lx2),int(ly2)),col,2)
        cv2.imwrite(os.path.join(args.debug_dir, f"debug_{tid}.jpg"), dbg)
    all_lines=sorted(all_lines, key=lambda x: x["length"], reverse=True)
    with open(args.output,"w",encoding="utf-8") as f: json.dump({"image":args.image,"image_size":{"W":W,"H":H,"origin":"top-left"},"tiling":args.tiles,"lines_count":len(all_lines),"lines":all_lines}, f, ensure_ascii=False, indent=2)
    print(f"\nГотово! {len(all_lines)} отрезков (будет много цифр как просил)")
    from collections import Counter; print(Counter(x["type"] for x in all_lines))
    csv=Path(args.output).with_suffix(".csv")
    with open(csv,"w",encoding="utf-8") as f:
        f.write("x1,y1,x2,y2,length,angle,type,tile\n")
        for l in all_lines: f.write(f"{l['x1']},{l['y1']},{l['x2']},{l['y2']},{l['length']},{l['angle']},{l['type']},{l['tile']}\n")
    print(f"CSV: {csv}")

if __name__=="__main__": main()
