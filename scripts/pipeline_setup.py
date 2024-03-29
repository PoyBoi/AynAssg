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


def pipelineSetup(model_path:int, prompt:str, clip_skip:int = 1):
    # text_encoder = os.path.join(model_path, "text_encoder")
    # print(text_encoder)
    model_path = model_path.split("\\")
    model_path = ["." if element == "AynAssg" else element for element in model_path]
    model_path = "/".join(model_path) + '/'


    if clip_skip > 1:
        text_encoder = transformers.CLIPTextModel.from_pretrained(
            model_path,
            subfolder = "text_encoder",
            num_hidden_layers = 12 - (clip_skip - 1),
            torch_dtype = torch_dtype
        )

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
            safety_checker = None,
            # repo_type = 
        )

    pipe = pipe.to(device_name)

    # Change the pipe scheduler to EADS.
    pipe.scheduler = diffusers.EulerAncestralDiscreteScheduler.from_config(
        pipe.scheduler.config
)