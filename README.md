# Ayna Assignment

</hr>

Experimentation pipeline for generating a 2048 x 2048 image from a text prompt describing a person and their background, emphasizing photorealism, steerability, and resource/time efficiency.


# Path to follow:

This is the path to follow:

1. [✅] Deploy Stable Diffusion into python, use [this link](https://medium.com/@natsunoyuki/using-civitai-models-with-diffusers-package-45e0c475a67e)
2. [✅] Convert civit.ai model required into diffuser model, using [this](https://github.com/huggingface/diffusers/blob/main/scripts/convert_original_stable_diffusion_to_diffusers.py), colab link pvt [here](https://colab.research.google.com/drive/1f8S3fCM9iDL7sk2Ny6gdvEiMs9-oO523#scrollTo=3NnPOMAqAABv)
    - for the same, try making a "mix" that can work with the problem statement given, using the "voldemort mix thing from 4chan", the "blends"
    - test the model, and see what iteration count works the best, read the description of each model correctly for the usage
3. Now apply the rest of the things:
    - Hypernetworks / ControlNet / Lora-Lycrosis / assign VAE (have to edit the pipeline for this) / CLIP skip
    - image upscaling / hi-res fix / assign height-width / face restoration
    - [✅] sampling steps (check model for best usage) / sampling method / cfg scale (maybe) / batch count
    - [✅] positive prompt / negative prompt / carried over prompt from what the model author tells / cfg / seed
    - [✅] save location / show folder / show seed / save prompt with image
4. Make it so that the image is generated in 512x512
    - make the model such that it has good realism, good geography, and good human relevance carry-over
    - copy the posture from the image
      - either use a lower CFG, or add a controlNet that extracts the pose from the image
      - or just use inpainting via segmentation of Unet
    - upscale the image from 512^2 to 2048^2
    - restore the faces if any
    - (Optional) Enhance the image using some method

# Links

1. YT [Link](https://www.youtube.com/watch?v=mZjrfN1SXXs) for the same
2. Form [Link](https://docs.google.com/forms/d/e/1FAIpQLSddT4uqrG3XJ6UnI_FScmG5N9TFLUFY0Ud4tMfLr_g6HnmZQg/viewform?pli=1)
