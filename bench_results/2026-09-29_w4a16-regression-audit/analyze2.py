import math, re, csv
K=32
CFG={'V1':('V1',8,8,2048),'A':('A',16,0,1024),'C':('C',16,0,512)}
def cfg(m,n,k):
    if m>256: return 'A' if n>=4096 else 'C'
    if m<4 and n>4096: return 'A'
    if 4<=m<=8 and n>=2048 and k>=1024: return 'C'
    if m==12 and 2560<=n<=8192 and k>=512: return 'C'
    if m==32 and 5120<=n<=6144 and k>=1536: return 'C'
    return 'V1'
def lds(c,k,s):
    _,mt,pad,_=CFG[c]; kps=k//s; return mt*(((kps+pad+7)//8)*8)*2
def budget(c,m,n):
    _,mt,_,nt=CFG[c]; b=math.ceil(m/mt)*math.ceil(n/nt)
    return (16*1024 if b>1024 else 64*1024 if b>256 else 32*1024), b
def legacy(c,m,n,k):
    bud,b=budget(c,m,n); mx=k//K; s=1
    while s<mx and lds(c,k,s)>bud: s*=2
    while s<16 and mx>=s*2 and (b*s<2048 or k//s>2048):
        cd=s*2
        if lds(c,k,cd)>bud: break
        s=cd
    return s
def enum(c,m,n,k):
    bud,b=budget(c,m,n); sp=[s for s in range(1,17) if k%s==0 and (k//s)%K==0]; i=0
    while i+1<len(sp) and lds(c,k,sp[i])>bud: i+=1
    while i+1<len(sp) and (b*sp[i]<2048 or k//sp[i]>2048):
        if lds(c,k,sp[i+1])>bud: break
        i+=1
    return sp[i]
def valid(k,s): return k%s==0 and (k//s)%K==0
# ground-truth fix splits from the fixed kernel debug log
fix={}
for line in open("fixed_build2_splits.log"):
    mm=re.search(r"\[rdna2_prefill_split\] m=(\d+) n=(\d+) k=(\d+) split=(\d+)", line)
    if mm: fix[(int(mm.group(1)),int(mm.group(3)),int(mm.group(2)))]=int(mm.group(4))  # (m,k,n)
def load(fn):
    d={}
    for row in csv.DictReader(open(fn)):
        if row["ms"]=="ERROR": continue
        d[(int(row["m"]),int(row["k"]),int(row["n"]),row["variant"])]=float(row["ms"])
    return d
old=load("old_build_prefill_timing.csv"); fixms=load("fixed_build2_prefill_timing.csv"); new=load("new_prefill_timing.csv")
def hasnan(fn):
    s=set()
    for row in csv.DictReader(open(fn)):
        try:
            if int(float(row["nan"]))==1: s.add((int(row["m"]),int(row["k"]),int(row["n"]),row["variant"]))
        except: pass
    return s
oldnan=hasnan("old_build_prefill_timing.csv"); newnan=hasnan("new_prefill_timing.csv")
shapes=[(m,k,n) for m in (225,2001,2048) for k in (1536,4352,5120,6144,8704) for n in (3584,4096,5120,8704)]
print(f"{'m':>5}{'k':>6}{'n':>6} {'cfg':>3} {'old':>3} {'new':>3} {'fix':>3} | {'oldms':>8}{'fixms':>8}{'d%':>6} | {'newms':>8}{'d%nw':>6} | onan nnan | note")
retunes=[]; repairs=[]
for (m,k,n) in shapes:
    c=cfg(m,n,k); o=legacy(c,m,n,k); nw=enum(c,m,n,k); fx=fix.get((m,k,n),-1)
    oms=old.get((m,k,n,'gptq')); fms=fixms.get((m,k,n,'gptq')); nms=new.get((m,k,n,'gptq'))
    onan=(m,k,n,'gptq') in oldnan; nnan=(m,k,n,'gptq') in newnan
    d=(fms-oms)/oms*100 if oms and fms else float('nan')
    dn=(fms-nms)/nms*100 if nms and fms else float('nan')
    note=""
    if not valid(k,o): note="OLD-INVALID->repair"; repairs.append((m,k,n,o,fx))
    elif nw!=o: note="RETUNE-REVERTED"; retunes.append((m,k,n,o,nw,fx))
    print(f"{m:>5}{k:>6}{n:>6} {c:>3} {o:>3} {nw:>3} {fx:>3} | {oms if oms else 0:>8.4f}{fms if fms else 0:>8.4f}{d:>6.2f} | {nms if nms else 0:>8.4f}{dn:>6.2f} | {str(onan)[0]}    {str(nnan)[0]}    | {note}")
print()
print("REPAIRS (old invalid):", repairs)
print("RETUNES (old valid, reverted):", retunes)
