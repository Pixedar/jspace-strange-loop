"""Native unquantized Thinker setup for the free-T4 pilot. Model weights remain upstream."""
import os,sys,json,time
from pathlib import Path
import numpy as np
import torch
from transformers import Qwen2_5OmniThinkerForConditionalGeneration,Qwen2_5OmniProcessor
MODEL_ID='Qwen/Qwen2.5-Omni-3B'
REV='f75b40e3da2003cdd6e1829b1f420ca70797c34e'
processor=Qwen2_5OmniProcessor.from_pretrained(MODEL_ID,revision=REV,min_pixels=112*112,max_pixels=112*112)
model=Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(MODEL_ID,revision=REV,dtype=torch.float16,device_map='cuda',attn_implementation='eager').eval()
for parameter in model.parameters():parameter.requires_grad_(False)
FOLLOWUP_ROOT=Path('followups')
print('READY',REV,torch.cuda.get_device_name(0),sum(p.numel() for p in model.parameters()))
