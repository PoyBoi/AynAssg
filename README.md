# <div align="center"><b>Project Chimera</b></div>

## <div align="center"><b>Ayna Assignment</b></div>

</hr>

<div align="center">
Experimentation pipeline for generating a 2048 x 2048 image from a text prompt describing a person and their background, emphasizing photorealism, steerability, and resource/time efficiency.
</div>

# :wrench: Dependencies and Installation

- Python >= 3.7 
- [PyTorch >= 1.7](https://pytorch.org/)
- Option: NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads) 

# [🤗](https://huggingface.co/) Installation

Following is the method to install this repo and get it up and working

1. Clone this repo on your local machine/cloud machine, anywhere
```
clone https://github.com/PoyBoi/AynAssg.git
cd AynAssg
```
2. Run a dry-run, it will run through the code and install the dependencies required
```
python main.py --r
```

3. Download your favourite models from your favourite repository collection, and place them in the `AynAssg/models/diffused` folder
    - I use [Civit.AI](https://civitai.com/)'s models as they are community backed and tested

# :rocket: Usage

Following are the usable methods as of now (will update in future if needed):
1. Convert ```.safetensor``` into a diffuser model to use with this repo
2. Generate images using any converted model
3. Change the background of an image with assisted inpainting and prompts
4. Upscale the image using Real-ESRGAN
5. Fix the faces in the image using GFPGAN

#### Note:
Please run this command to make sure you're inside the repo before running any of the commands
```
cd AynAssg
```

Here is how to run these methods:
## 🤖 Conversion
```
python main.py \
--c \
--l <Location of model>
```
#### Note:
> Model is stored in ```AynAssg\models\diffused``` within the folder of the same name as the original model


## 🖌️ Generation
```
python main.py \
--g \
--l <Location of model> \
--p '<Prompt, separated by commas>' \
--n '<Negative Prompt>' \
-batch-size <int> -steps <int> -size <w h> \
-seed <int> -cfg <int> -clip-skip <int> 
```
#### Note:
> Images are stored in ```AynAssg\outputs``` with prefix ```output_gen```


## 🖼️ Background Change
```
python main.py \
--b \
--l <Location of model> \
--p '<Prompt>' \
--n '<Negative Prompt>' \
-f '<Location of image>' \
-steps <int> -seed <int> -cfg <int> -clip-skip <int>
```
#### Note:
> Images are stored in ```AynAssg\outputs``` with prefix ```output_bg```


## 📈 Upscale and 👨👩 Face Restoration
```
python main.py \
--u <Upscaling's Scale> \
-f '<Location of image>'
```
#### Note:
> 1. Restored images are stored in ```AynAssg\results\restored_imgs```
> 2. Comparisions, cropped faces and restored faces are stores in their respective folders inside ```AynAssg\results```


## :bulb: Tips and Tricks
- Alter the sampling steps as per the sampler that you want to use, a choice will be given in the Terminal
    - When prompted, learn about the abbreviations if needed
- Feeling stuck ? Run this to find out about the methods you can use
```
python main.py -h
```
- A handy copy of the console
```
options:
  -h, --help                    show this help message and exit
  -convert, --c, -C             Check for if you want to convert a .safetensor model into a diffusor model and store it
  -generate, --g, -G            Sets mode to generate
  -background, --b, -B          Generates the background for an image
  -upscale U, --u U, -U U       Upscales the image by scale of <x>
  -setup, --r, -R               Does a dry run through the code, installing dependancies.
  -file F, --f F, -f F          Pass the location for the image to be used for inpainting
  -loc L, --l L, -L L           Set the location for the model
  -prompt P, --p P, -P P        Stores the prompt
  -neg-prompt N, --n N, -N N    Stores the negative prompt
  -seed S, --s S, -S S          Seed for generating the image
  -cfg CFG                      How imaginative the AI is, from a scale of 1 to
  -clip-skip CLIP_SKIP          Accounts for the CLIP skip setting
  -steps STEPS                  The amount of inference steps the models takes
  -batch-size BATCH_SIZE        Controls the number of images generated at once
  -size SIZE [SIZE ...]         Input the size of the image in W H
  -lora LORA                    Location of lora to be applied, if any
  ```
- Please have [cURL](https://curl.se/windows/) installed


## 🔃 Flow

### Generated Image
<p align="center">
  <img src="https://raw.githubusercontent.com/PoyBoi/AynAssg/main/outputs/miscHosted/miscHosted_gen.png">
  <br>
  Prompt: neon lights, female, cyberpunk, (wearing long coat, big collars), dark, cinematic, 8k uhd, realistic, looking at camera, portait
</p>

### Background Changed Image
<p align="center">
  <img src="https://raw.githubusercontent.com/PoyBoi/AynAssg/main/outputs/miscHosted/miscHosted_bg_1.png">
  <br>
  Prompt: (blurred, portait), neon, cyberpunk, background, realisitc, photshoot, alleyway, alley, japanese, 8k uhd
  <br><br><br>
  <img src="https://raw.githubusercontent.com/PoyBoi/AynAssg/main/outputs/miscHosted/miscHosted_bg_2.png">
  <br>
  Prompt: (blurred, portait), park, open air, trees, cyberpunk, sunset, beautiful,  background, realisitc, photshoot, 8k uhd
  <br><br><br>
  <img src="https://raw.githubusercontent.com/PoyBoi/AynAssg/main/outputs/miscHosted/miscHosted_bg_3.png">
  <br>
  Prompt: (blurred, portait), realisitc background of a beach shore on a sunset with waves and the ocean, photshoot, 8k uhd
  <br><br><br>
</p>

### Upscaled Image
<p align="center">
  <img src="https://raw.githubusercontent.com/PoyBoi/AynAssg/main/outputs/miscHosted/miscHosted_bg_3_up.png">
  <br>
  Upscaled from 512x512 to 2048x2048
  <br>
</p>




# Path followed:

This is the path to follow:

1. [✅] Deploy Stable Diffusion into python, use [this link](https://medium.com/@natsunoyuki/using-civitai-models-with-diffusers-package-45e0c475a67e)
2. [✅] Convert civit.ai model required into diffuser model, using [this](https://github.com/huggingface/diffusers/blob/main/scripts/convert_original_stable_diffusion_to_diffusers.py), colab link pvt [here](https://colab.research.google.com/drive/1f8S3fCM9iDL7sk2Ny6gdvEiMs9-oO523#scrollTo=3NnPOMAqAABv)
    - for the same, try making a "mix" that can work with the problem statement given, using the "voldemort mix thing from 4chan", the "blends"
    - test the model, and see what iteration count works the best, read the description of each model correctly for the usage
3. Now apply the rest of the things:
    - Hypernetworks / ControlNet / Lora-Lycrosis / assign VAE (have to edit the pipeline for this) / CLIP skip
    - [✅] image upscaling / hi-res fix / assign height-width / face restoration
    - [✅] sampling steps (check model for best usage) / sampling method / cfg scale (maybe) / batch count
    - [✅] positive prompt / negative prompt / carried over prompt from what the model author tells / cfg / seed
    - [✅] save location / show folder / show seed / save prompt with image
4. [✅] Make it so that the image is generated in 512x512
    - [✅] make the model such that it has good realism, good geography, and good human relevance carry-over
    - copy the posture from the image
      - either use a lower CFG, or add a controlNet that extracts the pose from the image
      - [✅] use inpainting via segmentation of Unet
    - [✅] upscale the image from 512^2 to 2048^2
    - [✅] restore the faces if any
    - [✅] Enhance the image using some method

# :warning: For the people
To see the path followed, refer
```
AynAssg/path.md
```

<!-- # Links

1. YT [Link](https://www.youtube.com/watch?v=mZjrfN1SXXs) for the same
2. Form [Link](https://docs.google.com/forms/d/e/1FAIpQLSddT4uqrG3XJ6UnI_FScmG5N9TFLUFY0Ud4tMfLr_g6HnmZQg/viewform?pli=1) -->
