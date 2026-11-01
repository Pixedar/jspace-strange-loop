"""Bounded live continuation and larger-backbone qualification.

Native Hugging Face full-context execution is a new generation mode. It uses
the qualified controller; it is not claimed identical to the old 256-token
sink-cache live runner. No weights of either LLM are fine-tuned.
"""
import os,json,pathlib,sys,time,gc,traceback,hashlib
os.environ.setdefault('HF_HOME',str(pathlib.Path(__file__).resolve().parents[1]/'cache'))
os.environ['TOKENIZERS_PARALLELISM']='false'
import numpy as np
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
ROOT=pathlib.Path(os.environ.get('RESEARCH_ROOT',pathlib.Path(__file__).resolve().parents[1])); sys.path.insert(0,str(ROOT/'src'))
from reader import Reader,Life
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32=False
OUT=ROOT/'results'; OUT.mkdir(exist_ok=True)
REVS={'Qwen/Qwen3-8B':'b968826d9c46dd6066d109eabc6255188de91218',
      'Qwen/Qwen3-14B':'40c069824f4251a91eefaf281ebe4c544efd3e18'}
def save(name,data):
    p=OUT/name; tmp=p.with_suffix(p.suffix+'.partial')
    tmp.write_text(json.dumps(data,indent=2)); tmp.replace(p)
def progress(stage,**kw):
    (ROOT/'progress.json').write_text(json.dumps(dict(time=time.time(),stage=stage,**kw)))
    print(stage,kw,flush=True)
def load_model(name):
    progress('load',model=name)
    tok=AutoTokenizer.from_pretrained(name,revision=REVS[name])
    model=AutoModelForCausalLM.from_pretrained(name,revision=REVS[name],torch_dtype=torch.bfloat16,
        device_map='cuda',attn_implementation='sdpa').eval().requires_grad_(False)
    return model,tok

class Coupled:
    def __init__(self,model,reader,snapshot,condition):
        self.model=model; self.reader=reader; self.life=Life(reader).restore(snapshot)
        self.life.config.update(exchange='swap' if condition=='schema_swap' else None,at=0,until=1000000)
        if condition=='unused_schema': self.life.config['schema']='none'
        self.condition=condition; self.content=None; self.history=None; self.scale=None
        self.norm_act=[];self.norm_delta=[];self.i=0;self.enabled=True
        self.hooks=[model.model.layers[16].register_forward_pre_hook(self.pre),
                    model.model.layers[16].register_forward_hook(self.postact),
                    model.model.layers[24].register_forward_hook(self.postread)]
    def pre(self,module,args):
        h=args[0]; self.enter=h.detach().float()
        if self.content is None or not self.enabled: return None
        act=self.content*self.scale if self.scale is not None else torch.zeros_like(self.content)
        if self.condition=='random_act':
            g=torch.Generator(device='cuda').manual_seed(6700+self.i)
            r=torch.randn(act.shape,device='cuda',generator=g)
            act=r*act.norm(dim=-1,keepdim=True)/r.norm(dim=-1,keepdim=True)
        allact=torch.cat([act,torch.zeros_like(act[:1])],0)[:,None]
        return (h+allact.to(h.dtype),)+args[1:]
    def postact(self,module,args,out):
        y=out[0] if isinstance(out,tuple) else out
        self.delta=(y.detach().float()-self.enter).norm(dim=-1).mean().item()
    def postread(self,module,args,out):
        self.read=(out[0] if isinstance(out,tuple) else out).detach().float()
    def update(self):
        self.history=self.enter[:4] if self.history is None else torch.cat([self.history,self.enter[:4]],1)[:,-16:]
        rec=self.life.step(self.read[:4,-1])
        l=self.history.shape[1]; hist=self.life.hist[:,-l:]
        key=self.reader.rk(self.reader.rln(hist)).view(4,l,4,64)
        q=self.reader.rq_bias.view(1,4,64).expand(4,-1,-1)
        baseline=torch.softmax(torch.einsum('nhd,nlhd->nhl',q,key)/8+self.reader.rpos[-l:][None,None],-1).mean(1)
        weights=rec['att'][:,-l:]-baseline
        self.content=torch.einsum('bl,bld->bd',weights,self.history)
        if self.i<32:
            self.norm_act.append(self.content.norm(dim=-1).mean().item()); self.norm_delta.append(self.delta)
        if self.i==31:
            self.scale=.5*float(np.median(self.norm_delta))/max(float(np.median(self.norm_act)),1e-8)
        self.i+=1
        return rec
    def close(self):
        for h in self.hooks:h.remove()

def sample(logits,u,temperature=.8,top_p=.9):
    p=torch.softmax(logits.float()/temperature,-1)
    v,ix=p.sort(descending=True)
    mask=v.cumsum(-1)-v>top_p; v[mask]=0; v=v/v.sum(-1,keepdim=True)
    j=(v.cumsum(-1)<u).sum(-1).clamp(max=v.shape[-1]-1)
    return ix.gather(1,j[:,None]).squeeze(1)

