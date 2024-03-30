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

# This should be the name of the main function
# def pipelineSetup(
def pipeCreate(
        model_path:str,
        clip_skip:int = 1
):
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

    # Adding this code block to pipelineSetup(), keeping here for testing
    # # Change the pipe scheduler to EADS.
    pipe.scheduler = diffusers.EulerAncestralDiscreteScheduler.from_config(
        pipe.scheduler.config
    )

    return pipe

def getEmbeddings(
    pipe:str,
    prompt:str,
    negative_prompt:str,
    max_length:int = None,
    device=torch.device("cpu")
)-> tuple:
    if max_length is None:
        max_length = pipe.tokenizer.model_max_length

    # Ensure prompt and negative prompt have the same length
    # assert len(prompt) == len(negative_prompt), "Prompt lists must have equal length"

    # Prepare inputs with padding and truncation
    inputs = pipe.tokenizer(
        prompt + negative_prompt,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length
    ).to(device)

    # Separate prompt and negative prompt encodings
    prompt_mask = torch.arange(len(prompt))[:, None].to(device)
    negative_mask = torch.arange(len(negative_prompt))[:, None].to(device) + len(prompt)

    prompt_embeddings = pipe.text_encoder(inputs)[prompt_mask]
    negative_embeddings = pipe.text_encoder(inputs)[negative_mask]

    return prompt_embeddings, negative_embeddings

#__main__
# This is the master function for this file
def pipelineSetup(
    model_path:str,
    # This might cause an issue
    # pipe:str,
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
    pipe = pipeCreate(model_path, clip_skip)

    # pipe.scheduler = diffusers.EulerAncestralDiscreteScheduler.from_config(
    # pipe.scheduler.config
    # )

    p_emd, n_emb = getEmbeddings(pipe, prompt, negative_prompt, max_length, device)

    # generation path code

    # Seed and batch size.
    # start_idx = 0
    # batch_size = 10
    # seeds = [i for i in range(start_idx , start_idx + batch_size, 1)]

    images = []

    # for count, seed in enumerate(seeds):
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
                prompt_embeds = p_emd,
                negative_prompt_embeds = n_emb,
                width = size[0],
                height = size[1],
                guidance_scale = cfg,
                num_inference_steps = steps,
                num_images_per_prompt = 1,
                generator = torch.manual_seed(seed),
            ).images


        images += new_img

        _, axs = plt.subplots(1, batch_size, figsize=(20, 20))

        # Plot each image in a subplot
        for i, img in enumerate(images):
            axs[i].imshow(img)
            axs[i].set_title(f'Image {i+1}')
            axs[i].axis('off')  # Hide axes

        plt.show()