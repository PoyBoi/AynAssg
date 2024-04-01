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

def pipeCreate(
        torch_dtype,
        model_path:str,
        clip_skip:int = 1
):
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
    return pipe

def getEmbeddings(
    pipe:str,
    prompt:str,
    negative_prompt:str,
    max_length:int = None,
    device=torch.device("cpu")
):
    if max_length is None:
        max_length = pipe.tokenizer.model_max_length

    # Ensure prompt and negative prompt have the same length        
    if len(prompt.split(",")) >= len(negative_prompt.split(",")):
        p_emb = pipe.tokenizer(
            prompt, return_tensors = "pt", truncation = False
        ).input_ids.to(device)
        shape_max_length = p_emb.shape[-1]
        n_emb = pipe.tokenizer(
            negative_prompt,
            truncation = False,
            padding = "max_length",
            max_length = shape_max_length,
            return_tensors = "pt"
        ).input_ids.to(device)

    # If negative prompt is longer than prompt.
    else:
        n_emb = pipe.tokenizer(
            negative_prompt, return_tensors = "pt", truncation = False
        ).input_ids.to(device)
        shape_max_length = n_emb.shape[-1]
        p_emb = pipe.tokenizer(
            prompt,
            return_tensors = "pt",
            truncation = False,
            padding = "max_length",
            max_length = shape_max_length
        ).input_ids.to(device)

    concat_embeds = []
    neg_embeds = []
    for i in range(0, shape_max_length, max_length):
        concat_embeds.append(
            pipe.text_encoder(p_emb[:, i: i + max_length])[0]
        )
        neg_embeds.append(
            pipe.text_encoder(n_emb[:, i: i + max_length])[0]
        )

    return torch.cat(concat_embeds, dim = 1), torch.cat(neg_embeds, dim = 1)

def plot_images(images, labels = None):
    N = len(images)
    n_cols = 3
    n_rows = int(np.ceil(N / n_cols))

    plt.figure(figsize = (20, 10))
    for i in range(len(images)):
        plt.subplot(n_rows, n_cols, i + 1)
        if labels is not None:
            plt.title(labels[i])
        plt.imshow(np.array(images[i]))
        plt.axis(False)
    plt.show()


#__main__
def pipelineSetup(
    model_path:str,
    prompt:str,
    negative_prompt:str,

    device=torch.device("cpu"),

    cfg:int = 7,
    clip_skip:int = 1,
    steps:int = 20,
    seed:int = -1,
    batch_size:int = 1,

    use_embeddings:bool = True,
    max_length:int = None,
    size:list = [512, 512],    
):
    if torch.cuda.is_available():
        device_name = torch.device("cuda")
        device = torch.device("cuda")
        torch_dtype = torch.float16
    else:
        device_name = torch.device("cpu")
        device = torch.device("cpu")
        torch_dtype = torch.float32

    pipe = pipeCreate(
        torch_dtype, 
        model_path = model_path, 
        clip_skip = clip_skip
        )

    pipe = pipe.to(device_name)

    if os.name == 'nt':  # Windows
        os.system('cls')
    else:  # Unix-like systems (macOS, Linux)
        os.system('clear')

    scheduler_classes = {
    1: diffusers.EulerDiscreteScheduler,
    2: diffusers.EulerAncestralDiscreteScheduler,
    3: diffusers.LMSDiscreteScheduler,
    4: diffusers.HeunDiscreteScheduler,
    5: diffusers.DPMSolverMultistepScheduler,
    9: diffusers.DPMSolverSinglestepScheduler,
    10: diffusers.KDPM2DiscreteScheduler,
    }

    diffusorOption = int(input(
        """

Which sampler do you want to use ?
1.  Euler               [Very simple and fast to compute but accrues error quickly unless a large number of steps (=small step size) is used.]
2.  Euler A             ["]
3.  LMS                 [A Linear Multi-Step method. An improvement over Euler's method that uses several prior steps, not just one, to predict the next sample.]
4.  Heun                [uses a correction step to reduce error and is thus an example of a predictor--corrector algorithm. Roughly twice as slow than Euler, not really worth using IME.]
5.  DPM++ 2M            [Variants of DPM++ that use second-order derivatives. Slower but more accurate. S means single-step, M means multi-step. DPM++ 2M (Karras) is probably one of the best samplers at the moment when it comes to speed and quality.]
6.  DPM++ 2M Karras     ["]
7.  DPM++ 2M SDE        ["]
8.  DPM++ 2M SDE Karras ["]
9.  DPM++ SDE           ["]
10. DPM2                [Diffusion Probabilistic Model solver. An algorithm specifically designed for solving diffusion differential equations, published by Cheng Lu et al.]

0. Explanation of abbreviations

hint-0: Select the Sampling Steps based on the Sampler, it changes how the output generates

Select your option:
"""
    )
)
    if diffusorOption == 0:
        print("""
What the abbv's mean:

? Any sampler with "Karras" in the name
= A noise schedule is essentially a curve that determines how large each diffusion step is. Works well in large steps at first and small steps at the end.

? Any sampler with "a" in the name
= An "ancestral" variant of the solver. The results are also usually quite different from the non-ancestral counterpart, often regarded as more "creative".

? Any sampler with "SDE" in the name
= Introduces some random "drift" to the process on each step to possibly find a route to a better solution than a fully deterministic solver. Doesn't necessarily converge on a single solution as the number of steps grow.

""")

    elif diffusorOption in scheduler_classes:
        pipe.scheduler = scheduler_classes[diffusorOption].from_config(pipe.scheduler.config)
    
    elif diffusorOption == 6:
        pipe.scheduler = diffusers.DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.scheduler.use_karras_sigmas = True

    elif diffusorOption == 7:
        pipe.scheduler = diffusers.DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.algorithm_type = "sde-dpmsolver++"

    elif diffusorOption == 8:
        pipe.scheduler = diffusers.DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.scheduler.use_karras_sigmas = True
        pipe.algorithm_type = "sde-dpmsolver++"


    else:
        raise ValueError(f"Invalid diffusorOption: {diffusorOption}")

    p_emb, n_emb = getEmbeddings(
        pipe, 
        prompt, 
        negative_prompt, 
        max_length, 
        device,
        )

    images, labelsImg = [], []

    for count in range(batch_size):
        # start_time = time.time()
        if use_embeddings is False:
            new_img = pipe(
                prompt = prompt,
                negative_prompt = negative_prompt,
                width = size[0],
                height = size[1],
                guidance_scale = cfg,
                num_inference_steps = steps,
                num_images_per_prompt = 1,
                generator = torch.manual_seed(seed),
            ).images
        else:
            new_img = pipe(
                prompt_embeds = p_emb,
                negative_prompt_embeds = n_emb,
                width = size[0],
                height = size[1],
                guidance_scale = cfg,
                num_inference_steps = steps,
                num_images_per_prompt = 1,
                generator = torch.manual_seed(seed),
            ).images


        images += new_img
        labelsImg += "Prompts: {} | Negative Prompts: {} | CFG: {} | Size: {}".format(
            prompt,
            negative_prompt, 
            cfg, 
            size
        )

    plot_images(images, labelsImg)