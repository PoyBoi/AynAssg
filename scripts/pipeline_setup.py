import argparse
import diffusers
import transformers

import os
import sys
import time
import shutil
from datetime import datetime

import torch
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

import cv2
from torchvision import transforms
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image, make_image_grid


now = datetime.now()
timestamp = now.strftime("%Y%m%d_%H%M%S")


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
            padding = 'max_length',
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
            padding = 'max_length',
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
    n_cols = 5
    n_rows = int(np.ceil(N / n_cols))

    plt.figure(figsize = (20, 5 * n_rows))
    for i in range(len(images)):
        plt.subplot(n_rows, n_cols, i + 1)
        if labels is not None:
            plt.title(labels[i])
        plt.imshow(np.array(images[i]))
        plt.axis(False)
    plt.show()

def swapBG(
        model_path:str,
        prompt_bg:str,
        negative_prompt_bg:str,

        filename:str,
        
        seed:int=-1,
        steps:int = 20,
        cfg:int = 7
):
    
    model_path = model_path.split("\\")
    model_path = ["." if element == "AynAssg" else element for element in model_path]
    model_path = "/".join(model_path) + '/'
    
    if torch.cuda.is_available():
        device_name = torch.device("cuda")
        device = torch.device("cuda")
        torch_dtype = torch.float16
    else:
        device_name = torch.device("cpu")
        device = torch.device("cpu")
        torch_dtype = torch.float32

    model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True)
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet101', pretrained=True)
    model.eval()

    input_image = Image.open(filename)
    input_image = input_image.convert("RGB")
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)

    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    with torch.no_grad():
        output = model(input_batch)['out'][0]
    output_predictions = output.argmax(0)

    palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
    colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
    colors = (colors % 255).numpy().astype("uint8")

    r = Image.fromarray(output_predictions.byte().cpu().numpy()).resize(input_image.size)
    r.putpalette(colors)

    save_path = r"./outputs/tmp/tmp_mask_0.png"
    r.save(save_path)

    plt.imshow(r)
    plt.show()

    image = cv2.imread(r"./outputs/tmp/tmp_mask_0.png")
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    threshold = 0

    # Convert to black and white using manual threshold with explicit black and white values
    black_white_image = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY)[1]
    inverted_image = cv2.bitwise_not(black_white_image)

    cv2.imwrite(r'./outputs/tmp/tmp_mask.png', inverted_image)

    pipeline = AutoPipelineForInpainting.from_pretrained(
        model_path,
        torch_dtype=torch.float16, variant="fp16", safety_checker = None, requires_safety_checker = False,
    )

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
        pipeline.scheduler = scheduler_classes[diffusorOption].from_config(pipeline.scheduler.config)
    
    elif diffusorOption == 6:
        pipeline.scheduler = diffusers.DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
        pipeline.scheduler.use_karras_sigmas = True

    elif diffusorOption == 7:
        pipeline.scheduler = diffusers.DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
        pipeline.algorithm_type = "sde-dpmsolver++"

    elif diffusorOption == 8:
        pipeline.scheduler = diffusers.DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
        pipeline.scheduler.use_karras_sigmas = True
        pipeline.algorithm_type = "sde-dpmsolver++"


    else:
        raise ValueError(f"Invalid diffusorOption: {diffusorOption}")

    pipeline.enable_model_cpu_offload()

    # load base and mask image
    init_image = load_image(filename)
    mask_image = load_image(r'./outputs/tmp/tmp_mask.png')

    # 92 -> seed
    generator = torch.Generator("cuda").manual_seed(seed)
    # prompt = prompt_bg
    # negative_prompt = negative_prompt_bg

    print(sum(init_image.size), sum(mask_image.size))

    if init_image.size == mask_image.size:
        print(sum(init_image.size), sum(mask_image.size))
        # if (sum(init_image.size) + sum(mask_image.size)) > 1999:
        image_1 = pipeline(
            prompt=prompt_bg, 
            negative_prompt=negative_prompt_bg, 
            image=init_image, mask_image=mask_image, generator=generator, 
            width = init_image.size[0], height = init_image.size[1],
            num_inference_steps= steps,
            guidance_scale=cfg
        ).images
    # make_image_grid([init_image, mask_image, image], rows=1, cols=3)

    # print(image_1, "\n", image_1[0])
    image_1[0].show()

    # # Get the current date and time
    # now = datetime.now()

    # # Format the date and time as a string
    # timestamp = now.strftime("%Y%m%d_%H%M%S")

    # Use the timestamp in the filename
    filename = f"output_bg_{timestamp}.png"
    image_1[0].save("./outputs/" + filename)

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

    images, labelsImg = [], []

    for count in range(batch_size):

        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")

        if seed == -1:
            seed = timestamp
            old_ts = timestamp
        if seed == old_ts:
            seed = timestamp
            old_ts = timestamp
        
        if use_embeddings is False:
            new_img = pipe(
                prompt = prompt,
                negative_prompt = negative_prompt,
                width = int(size[0]),
                height = int(size[1]),
                guidance_scale = cfg,
                num_inference_steps = steps,
                num_images_per_prompt = 1,
                generator = torch.manual_seed(seed),
            ).images
        else:
            p_emb, n_emb = getEmbeddings(
                        pipe, 
                        prompt, 
                        negative_prompt, 
                        max_length, 
                        device,
                        )
            new_img = pipe(
                prompt_embeds = p_emb,
                negative_prompt_embeds = n_emb,
                width = int(size[0]),
                height = int(size[1]),
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

        filename = f"output_gen_{timestamp}.png"
        new_img[0].save("./outputs/" + filename)

    for i in images:
        i.show()