import os
import argparse
import subprocess

from scripts.pipeline_setup import (pipelineSetup, swapBG)

parser = argparse.ArgumentParser()
parser.add_argument('-convert', '--c', '-C', help='Check for if you want to convert a .safetensor model into a diffusor model and store it', action='store_true')
parser.add_argument('-generate', '--g', '-G', help='Sets mode to generate', action='store_true')
parser.add_argument('-background', '--b', '-B', help='Generates the background for an image', action='store_true')

parser.add_argument('-file', '--f', '-f', help='Pass the location for the image to be used for inpainting', default='')

parser.add_argument('-loc', '--l', '-L', type=str, help='Set the location for the model', default='')
parser.add_argument('-prompt', '--p', '-P', type=str, help='Stores the prompt', default='')
parser.add_argument('-neg-prompt', '--n', '-N', type=str, help='Stores the negative prompt', default='')

parser.add_argument('-seed', '--s', '-S', type=int, help='Seed for generating the image', default=-1)
parser.add_argument('-cfg', type=int, help='How imaginative the AI is, from a scale of 1 to ', default=7)
parser.add_argument('-clip-skip', type=int, help='Accounts for the CLIP skip setting', default=1)
parser.add_argument('-steps', type=int, help='The amount of inference steps the models takes', default=20)
parser.add_argument('-batch-size', type=int, help='Controls the number of images generated at once', default=1)

parser.add_argument('-size', nargs='+', help='Input the size of the image in W H', default=[512, 512])

parser.add_argument('-lora', type=str, help="Location of lora to be applied, if any", default='')

args = parser.parse_args()

if os.name == 'nt':  # Windows
    os.system('cls')
else:  # Unix-like systems (macOS, Linux)
    os.system('clear')

print("---> Fetching CWD")
current_dir = os.path.dirname(os.path.realpath(__file__))
print("->", current_dir, "\n")



print("---> Setting CWD")
command = "cd {}".format(current_dir)
process = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
output, error = process.communicate()
print("-> Done \n")



print("---> Updating Requirements")
command = r"pip install -r requirements.txt"
process = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
output, error = process.communicate()
print("-> Done \n")


#__main__

# Sample input line:
# python main.py --c --l C:\TheGoodShit\StableDiffusion\stable-diffusion-webui\models\Stable-diffusion\beautifulRealistic_v60.safetensors

if args.l !='':
    loc = args.l
    file_name = loc.split("\\")[-1].split(".")[0]
    
    # Removes incorrect literals
    noUse = ['-', '_', '.', '--', '..']
    for i in file_name:
        if i in noUse:
            file_name_new = file_name.split(i)[0]
            file_name = file_name_new
            break

    print("\n------> Look for the model inside: models\diffused\{}\n".format(file_name))

    # Model conversion pipeline
    if args.c == True:
        print("Converting {} to diffusor based model".format(file_name))
        # os.makedirs("models\diffused\{}".format(file_name), exist_ok=True)

        command = "python models\convertModel.py  --checkpoint_path {} --dump_path models\diffused\{} --from_safetensors".format(
            loc, file_name
        )

        process = subprocess.Popen(
            command, 
            stdout=subprocess.PIPE, 
            shell=True
        )
        
        output, error = process.communicate()
        print("Diffusor stored at \models\diffused\{}".format(file_name))
    
    # Generation pipeline
    if args.g == True:
        # print(args.size)
        pipelineSetup(
            model_path = args.l, 
            prompt = args.p,
            negative_prompt = args.n,
            cfg = args.cfg,
            clip_skip = args.clip_skip,
            steps = args.steps,
            seed = args.s, 
            batch_size = args.batch_size,
            size = args.size            
            )
    
    if args.b == True:
        print("IN !")
        if os.path.isfile(args.f):
            swapBG(
                model_path = args.l, 
                prompt_bg = args.p,
                negative_prompt_bg = args.n,
                filename = args.f,
                cfg = args.cfg,
                steps = args.steps,
                seed = args.s,
                # clip_skip = args.clip_skip,
                # batch_size = args.batch_size,
                # size = args.size  
            )
        else:
            print("[Location Error] Path to/and/or Image do(es) not exist")

    else:
        print("[Generation/Conversion Error] No correct method selected, use '-h' to get list of available methods to use")
else:
    print("[Location Error] Location of model left empty, use '-h' to get list of available methods to use")