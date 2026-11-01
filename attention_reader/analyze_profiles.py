"""Recompute life-level organization from the optional numeric evidence package."""
import argparse,json,pathlib,numpy as np
def measures(L,first=3,gap=3):
    L=L.astype(np.float64);n,b,d=L.shape
    V=L.reshape(-1,d);V-=V.mean(1,keepdims=True);V/=np.maximum(np.linalg.norm(V,axis=1,keepdims=True),1e-12)
    C=(V@V.T).reshape(n,b,n,b);own=[];other=[]
    for i in range(first,n):
      for j in range(i+gap,n):
       for c in range(b):
        own.append(C[i,c,j,c]);other.extend(C[i,c,j,e] for e in range(b) if e!=c)
    return float(np.mean(own)-np.mean(other))
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',required=True);p.add_argument('--index',required=True);p.add_argument('--out',required=True);args=p.parse_args()
    data=np.load(args.data);index=json.loads(pathlib.Path(args.index).read_text());rows=[]
    for e in index:rows.append(dict(**e,persistence=measures(data[e['key']],e['first_segment'],e['gap_segments'])))
    out=pathlib.Path(args.out);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(rows,indent=2));print('recomputed',len(rows),'life/condition profiles')
if __name__=='__main__':main()
