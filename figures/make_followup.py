"""Regenerate the multimodal follow-up figure from published measurements."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parents[1]
results=root/'followups/results/multimodal'
screen=json.loads((results/'intake_screening.json').read_text())['results']
modal=json.loads((results/'modality_check.json').read_text())['rows']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'text.color':'#24313b','axes.labelcolor':'#24313b','axes.edgecolor':'#d6dce1','xtick.color':'#687785','ytick.color':'#687785'})
fig,axes=plt.subplots(1,2,figsize=(12,4.5),gridspec_kw={'width_ratios':[1,1.15]})
ax=axes[0]
for i,intake in enumerate(('suffix','balanced')):
    for j,condition in enumerate(('self','recorded_other')):
        values=[r['heldout_colours']['state_dependent_share'] for r in screen if r['intake']==intake and r['condition']==condition]
        x=i+(.17 if j else -.17)
        ax.bar(x,np.mean(values),width=.30,color=['#287d76','#a4adb5'][j],label=['Self-description','Recorded-other control'][j] if i==0 else None)
        ax.scatter([x-.04,x+.04],values,s=23,color='#24313b',zorder=4)
ax.set_xticks([0,1],['Recent text','Image + audio + text']);ax.set_ylim(0,.61)
ax.set_ylabel('Normalized state dependence');ax.set_title('Access changes what can develop',weight='bold',pad=16)
ax.legend(frameon=False,fontsize=9,loc='upper left')
ax.text(.02,.02,'Points: two seeds · held-out colors',transform=ax.transAxes,size=9,color='#687785')
ax=axes[1]
keys=['image_reversal_destination_rms','spoken_direction_destination_rms','factorial_destination_interaction_rms']
for seed,offset,c in [(0,-.17,'#287d76'),(1,.17,'#426b9e')]:
    rows=[r for r in modal if r['condition']=='self' and r['seed']==seed]
    values=[np.mean([r[k] for r in rows]) for k in keys]
    ax.bar(np.arange(3)+offset,values,width=.30,color=c,label=f'Seed {seed}')
    for x,v in zip(np.arange(3)+offset,values):ax.text(x,v*1.18,f'{v:.3f}',ha='center',size=8.5)
ax.set_yscale('log');ax.set_ylim(.001,2)
ax.set_xticks([0,1,2],['Reverse image','Swap speech','Nonadditive\ninteraction'])
ax.set_ylabel('Change in next destination (RMS)');ax.set_title('The rich loop is mainly driven by speech',weight='bold',pad=16)
ax.legend(frameon=False,fontsize=9,loc='upper left')
for ax in axes:ax.spines[['top','right']].set_visible(False)
fig.subplots_adjust(wspace=.35,bottom=.19)
fig.text(.5,-.005,'Cached native-state screen, not full live spotlight training. Four training combinations; four held-out-color combinations.\nInput-intervention bars average both palettes. Nonzero interaction alone does not establish a unified representation.',ha='center',size=9,color='#687785')
for suffix in ('png','svg'):fig.savefig(root/f'figures/multimodal_followup.{suffix}',dpi=180,bbox_inches='tight',pad_inches=.18,facecolor='white')
print('Multimodal follow-up figure regenerated.')
