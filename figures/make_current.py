"""Generate publication figures and compact interactive data from saved numeric records."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import torch

STAGE=Path(__file__).resolve().parents[1]
FIG=STAGE/'figures'
FIG.mkdir(exist_ok=True)
ROOT=STAGE/'self_reference/results/v24'
INK='#24313b'; MUTED='#687785'; TEAL='#287d76'; RED='#c6513a'; BLUE='#426b9e'; GREY='#a4adb5'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'text.color':INK,'axes.labelcolor':INK,
                     'axes.edgecolor':'#d6dce1','xtick.color':MUTED,'ytick.color':MUTED,'savefig.facecolor':'white'})


def load(path): return json.loads(path.read_text(encoding='utf-8'))


def finish(fig,name):
    fig.savefig(FIG/(name+'.png'),dpi=180,bbox_inches='tight',pad_inches=.18)
    fig.savefig(FIG/(name+'.svg'),bbox_inches='tight',pad_inches=.18)
    plt.close(fig)


def box(ax,x,y,w,h,title,body,colour=INK):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.01,rounding_size=.018',facecolor='white',edgecolor=colour,lw=1.5))
    ax.text(x+w/2,y+h*.67,title,ha='center',va='center',weight='bold',size=12,color=colour)
    ax.text(x+w/2,y+h*.30,body,ha='center',va='center',size=10,color=MUTED,linespacing=1.5)


fig,ax=plt.subplots(figsize=(12,6.3)); ax.set(xlim=(0,1),ylim=(0,1)); ax.axis('off')
ax.text(.02,.97,'The control state changes what the reader reads next',size=18,weight='bold',va='top')
ax.text(.02,.895,'The language model’s weights stay fixed. The reader, summary map, decoder and prediction network learn.',size=10.5,color=MUTED)
box(ax,.04,.52,.21,.18,'Language model','processes the text')
box(ax,.39,.52,.21,.18,'Reader','attends to recent activity',TEAL)
box(ax,.75,.52,.21,.18,'Summary','32 learned numbers',TEAL)
box(ax,.75,.18,.21,.18,'Control state','8 numbers set the next read',TEAL)
for x0,x1 in ((.25,.39),(.60,.75)):
    ax.add_patch(FancyArrowPatch((x0,.61),(x1,.61),arrowstyle='-|>',mutation_scale=14,color=TEAL,lw=1.8))
ax.text(.32,.65,'activity',ha='center',size=10,color=MUTED)
ax.add_patch(FancyArrowPatch((.855,.52),(.855,.36),arrowstyle='-|>',mutation_scale=14,color=TEAL,lw=1.8))
ax.plot([.75,.495,.495],[.27,.27,.52],color=TEAL,lw=1.8)
ax.add_patch(FancyArrowPatch((.495,.43),(.495,.52),arrowstyle='-|>',mutation_scale=14,color=TEAL,lw=1.8))
ax.text(.61,.30,'sets the reader’s query',ha='center',size=10,color=TEAL)
ax.plot([.495,.495,.145,.145],[.70,.79,.79,.70],color=RED,lw=1.8)
ax.add_patch(FancyArrowPatch((.145,.755),(.145,.70),arrowstyle='-|>',mutation_scale=14,color=RED,lw=1.8))
ax.text(.32,.81,'the same attention also focuses the model',ha='center',size=10,color=RED)
ax.text(.04,.40,'Shared attention bias',color=RED,weight='bold',size=11)
ax.text(.04,.355,'Changes the model’s predictions.\nIts effects return only weakly\nthrough the reader.',va='top',color=MUTED,size=10,linespacing=1.6)
ax.text(.02,.055,'During training, a prediction network uses the summary to describe responses from alternative starting states.\nThe training loss changes the loop. During operation, the summary decoder supplies the next state directly.',size=10.5,color=MUTED,linespacing=1.5)
finish(fig,'self_describing_attention')

# Actual probe-response arrows from one matched held-out token/context.
selected=['another_s1_spot','dev_s0_seed0_spot']
S=torch.randn(12,8,generator=torch.Generator().manual_seed(4242)); S=(S/S.norm(dim=-1,keepdim=True)).numpy()
dest=[np.load(ROOT/n/'fields_iface_3000.npz')['dest'][30,0] for n in selected]
points=np.concatenate([S]+dest); centre=points.mean(0); _,sv,V=np.linalg.svd(points-centre,full_matrices=False)
basis=V[:2].T
proj=lambda z:(z-centre)@basis
fig,axs=plt.subplots(1,2,figsize=(11.3,4.5),sharex=True,sharey=True)
for ax,n,D,title,c in zip(axs,selected,dest,['Describing another run','Describing its own dynamics'],[GREY,TEAL]):
    p,q=proj(S),proj(D)
    for i in range(8):
        ax.annotate('',xy=q[i],xytext=p[i],arrowprops=dict(arrowstyle='-|>',lw=1.25,color=c,alpha=.8,shrinkA=4,shrinkB=3))
    ax.scatter(p[:8,0],p[:8,1],s=26,facecolors='white',edgecolors=INK,zorder=3,label='current probe state')
    ax.scatter(q[:8,0],q[:8,1],s=25,color=c,zorder=4,label='measured next state')
    ax.set_title(title,pad=14,weight='bold'); ax.set_aspect('equal',adjustable='box')
    ax.spines[['top','right']].set_visible(False)
    ax.set_xlabel('Projection axis 1')
axs[0].set_ylabel('Projection axis 2'); axs[0].legend(frameon=False,fontsize=8.5,loc='upper left')
fig.suptitle('Changing the starting state can change where the loop goes next',size=15,weight='bold',y=1.03)
fig.text(.5,-.04,'Same text, token and probe states. Eight of 12 measured responses shown in a 2-dimensional projection.\nHeld-out token 600, copy 0; projection captures %.1f%% of the displayed point variance.'%(100*(sv[:2]**2).sum()/(sv**2).sum()),ha='center',size=9,color=MUTED)
finish(fig,'state_dependence')

rows=[]
for n in ['dev_s0_seed0_spot','dev_s0_seed2_spot','dev_s1_seed1_spot','rich_s0s2_spot','rich_s1s1_spot','another_s1_spot']:
    a=load(ROOT/n/'self_model_residual_3000.json')
    c=load(ROOT/n/'spotlight_check_3000.json')[n]
    rows.append({'id':n,'dependence':a['remainder_share_of_field'],'correlation':a['predicted_remainder_correlation'],
                 'unscaled_explained':a['predicted_remainder_explained'],'changed':c['against_no_channel']['own']['top_prediction_changed'],
                 'loss_pct':c['against_no_channel']['own']['loss_change_pct'],'model_route_ratio':c['field_routes']['spotlight']/c['field_routes']['both']})
labels=['Development A / seed 0','Development A / seed 2','Development B / seed 1','Continuation A','Continuation B','Other-run description']
fig,axs=plt.subplots(1,3,figsize=(13,4.7),gridspec_kw={'width_ratios':[1.3,1,1]})
y=np.arange(len(rows)); colours=[TEAL]*5+[GREY]
for ax,field,title,scale,form in zip(axs,['dependence','changed','loss_pct'],['State dependence','Top prediction changes','Next-token loss change'],[1,100,1],['.2f','.1f','.2f']):
    vals=np.array([r[field] for r in rows])*scale
    ax.barh(y,vals,color=colours,height=.6)
    for yi,val in zip(y,vals): ax.text(max(val,0)+(.012 if field=='dependence' else .22 if field=='changed' else .025),yi,format(val,'.3f' if field=='dependence' and val<.01 else form)+('%' if field!='dependence' else ''),va='center',size=9)
    ax.set_title(title,weight='bold',pad=14); ax.invert_yaxis(); ax.spines[['top','right','left']].set_visible(False)
    ax.set_yticks(y,labels if field=='dependence' else ['']*len(y)); ax.tick_params(axis='y',length=0)
    ax.axvline(0,color='#d6dce1',lw=.8)
    ax.set_xlabel('normalized squared variation' if field=='dependence' else '% of held-out tokens' if field=='changed' else '% relative to no channel')
    ax.set_xlim(right=(.83 if field=='dependence' else 20 if field=='changed' else 2.05))
fig.suptitle('The loop affects the model, but these are different measurements',size=15,weight='bold',y=1.03)
fig.text(.5,-.045,'Five self-describing runs and one recorded-other control; Qwen3-0.6B. State dependence is a ratio, not a probability.\nTeacher-forced prediction changes do not measure usefulness or a stable personality. Negative loss change is better.',ha='center',size=9,color=MUTED)
fig.tight_layout(w_pad=2); finish(fig,'consequences')

best=load(ROOT/'dev_s0_seed0_spot/self_formation.json')
fig,axs=plt.subplots(1,2,figsize=(10.8,4.2))
cuts=['reference','reading cut','processing cut']; titles=['Intact','Reader path cut','Model path cut']
vals=[100*best['jumps'][k]['jumps_over_half'] for k in cuts]
axs[0].bar(titles,vals,color=[TEAL,RED,GREY],width=.6)
axs[0].set_ylabel('Steps larger than 0.5 (%; unit control state)'); axs[0].set_ylim(0,34)
for i,v in enumerate(vals): axs[0].text(i,v+.8,f'{v:.1f}%',ha='center',size=10)
axs[0].set_title('The jumps depend on the reader path',weight='bold',pad=14)
ts=[1,2,5,10,20,40]
for k,label,c in [('foreign / full','Both paths',TEAL),('foreign / reading','Reader path only',BLUE),('foreign / processing','Model path only',GREY)]:
    axs[1].plot(ts,[best['picture_difference_remaining'][k][str(t)] for t in ts],marker='o',ms=4,lw=1.6,label=label,color=c)
axs[1].set(xlabel='Tokens after a picture transplant',ylabel='Picture difference remaining',ylim=(-.03,.76))
axs[1].set_title('A transplanted picture can persist',weight='bold',pad=14); axs[1].legend(frameon=False,fontsize=9)
for ax in axs: ax.spines[['top','right']].set_visible(False)
fig.text(.5,-.04,'Best developmental run, held-out texts. Transplant values are averages over selected intervention times and copies.\nThe two reader-path curves nearly overlap; persistence varies markedly across runs.',ha='center',size=9,color=MUTED)
fig.tight_layout(w_pad=3);finish(fig,'causal_paths')

# One common projection for a recorded trajectory comparison.
view_names=['dev_s0_seed0_spot','another_s1_spot','rich_s0s2_spot']
trajectories=[np.load(ROOT/n/'self_formation_ref.npz') for n in view_names]
all_s=np.concatenate([z['s'][200:1000].reshape(-1,8) for z in trajectories]); mean=all_s.mean(0)
_,sigma,vt=np.linalg.svd(all_s-mean,full_matrices=False); axes=vt[:2].T
viewer={'projection':{'kind':'PCA','dimension':8,'shown':2,'variance_explained':float((sigma[:2]**2).sum()/(sigma**2).sum()),'fit':'pooled selected runs, held-out steps 200–999'},
        'runs':[],'measurements':rows,'best_cuts':best['jumps'],'best_transplant':best['picture_difference_remaining'],
        'sources':['self_reference/results/v24/'+n+'/self_formation_ref.npz' for n in view_names]}
for n,z in zip(view_names,trajectories):
    p=(z['s'][200:1000]-mean)@axes
    viewer['runs'].append({'id':n,'label':{'dev_s0_seed0_spot':'Self-description: stable state-dependent run','another_s1_spot':'Recorded-other description: calm control','rich_s0s2_spot':'Self-description: oscillating continuation'}[n],
        'start_token':200,'xy':np.round(p,5).tolist(),'steps':np.round(np.linalg.norm(np.diff(z['s'][199:1000],axis=0),axis=-1),5).tolist(),
        'attention_entropy':np.round(z['att_entropy'][200:1000],5).tolist()})
(STAGE/'interactive').mkdir(exist_ok=True)
(STAGE/'interactive/trajectories.json').write_text(json.dumps(viewer,separators=(',',':')),encoding='utf-8')
(STAGE/'self_reference/results/verified_summary.json').write_text(json.dumps({'runs':rows,'metric_note':'State-dependence numerator is centered next-destination squared variation; denominator is total displacement-field variation. This is not a bounded variance decomposition or a fraction of steps.'},indent=2)+'\n',encoding='utf-8')
print('FIGURES',len(list(FIG.glob('*.png'))),'VIEWER_BYTES',(STAGE/'interactive/trajectories.json').stat().st_size,flush=True)
