import argparse
import diffusers
import transformers

import sys
import os
import shutil
import time

import torch
import matplotlib.pyplot as plt
import numpy as np

from PIL import Image

if torch.cuda.is_available():
    device_name = torch.device("cuda")
    torch_dtype = torch.float16
else:
    device_name = torch.device("cpu")
    torch_dtype = torch.float32


def pipelineSetup(model_path:int,clip_skip:int,):
    text_encoder = os.path.join(model_path, "text_encoder")
    if clip_skip != 1:
        # TODO clean this up with the condition below.
        pipe = diffusers.DiffusionPipeline.from_pretrained(
            model_path,
            torch_dtype = torch_dtype,
            safety_checker = None,
            text_encoder = text_encoder, 
        )
    else:
        pipe = diffusers.DiffusionPipeline.from_pretrained(
            model_path,
            torch_dtype = torch_dtype,
            safety_checker = None
        )