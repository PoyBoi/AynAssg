# The path I took:

1.  [x] Setup the repo to work in it
2.  [x] Turning the model into a diffuser model
3.  [x] Turned it into a pipeline
4.  [x] Loaded the diffusor model
 - Trying these models:
    - [ ] clarity_3
    - [ ] dreamshaper_8
    - [ ] [realisticVision_6](https://huggingface.co/SG161222/Realistic_Vision_V6.0_B1_noVAE)
5.  [x] Wrote argparser for CLI
6.  [x] Finish the path from the article
7.  [x] Need to fix the argparse in main.py
8.  [x] Need to add save location
9.  [ ] Add real-ESRGAN
10. [x] Error Handling
11. [ ] Lora
    > Try these:
    - negative embeddings lora's
12. [ ] Make a model selector menu that reads models from the diffused folder
13. [x] Rehashing the code:
    - [x] Check if the pipeline setup inside a function is an issue, should not be an issue because the embdedding one works, will have to see about the global effect of using EADS in it or not
    - [x] Debug the masks and the emb's 117-122
    - [x] Remove the commented code, shift learnings to a ignored file
    - The changes that I made:
        - [x] Took lines 167-170 from inside pipeCreate
        - [x] added .input_ids
        - [x] replaced getEmbdedding's main core logic, what a shame
14. [x] Fix the output screen
    - [ ] Show the prompt, the neg prompt, the seed, cfg scale
    - [ ] Load the above details into the pictures (check civit.ai)
15. [x] Show save location
    - [x] Make a save folder logic, checks for exist
    - [x] The name for the file saving
16. [x] Try adding a progress loading bar
17. [x] Check how to change the sampler
18. [ ] For the main task:
    - [ ] Need to make one model that gen's the BG and the other that gen's the FG
    - [x] Or I can use a singular model for this
        - [x] Use DeepLabv3+ for the mask
        - [x] Pass that to the BG generation model
        - [x] Assign the mask image to a variable
            - [x] Save the mask, name is temp_mask inside output
        - [ ] Dump memory to clear space in the end off the VRAM after gen is done
        - [ ] Add inpainting model option to argparse
            - [ ] Add BG prompt to the CLI as well
        - [ ] Make this a function that is called
        - [ ] Have to push to main pipeline_setup file
    - [ ] Or I can gen the FG, cut it out using UNet, and then gen a BG
19. [ ] Add a method that upscales the image
20. [ ] Try to add controlNet inpainting
    - [ ] Can add controlNet Aux for posture detection