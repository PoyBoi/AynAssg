import os
import argparse
import subprocess

from scripts.pipeline_setup import (pipelineSetup, swapBG)

parser = argparse.ArgumentParser()
parser.add_argument('-convert', '--c', '-C', help='Check for if you want to convert a .safetensor model into a diffusor model and store it', action='store_true')
parser.add_argument('-generate', '--g', '-G', help='Sets mode to generate', action='store_true')
parser.add_argument('-background', '--b', '-B', help='Generates the background for an image', action='store_true')
parser.add_argument('-upscale', '--u', '-U', type=int, help='Upscales the image by scale of <x>', default=0)

parser.add_argument('-file', '--f', '-f', help='Pass the location for the image to be used for inpainting', default='')

parser.add_argument('-loc', '--l', '-L', type=str, help='Set the location for the model', default='')
parser.add_argument('-prompt', '--p', '-P', type=str, help='Stores the prompt', default='')
parser.add_argument('-neg-prompt', '--n', '-N', type=str, help='Stores the negative prompt', default='')

parser.add_argument('-seed', '--s', '-S', type=int, help='Seed for generating the image', default = -1)
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


print("---> Checking for GFPGAN installation")
dir_path = "models/dependancy/GFPGAN"
if os.path.isdir(dir_path):
    print("-> Already installed")
    print("-> Checking for updates")
    subprocess.run(["git", "-C", dir_path, "fetch"])

    # Check the status of the local repository
    status = subprocess.check_output(["git", "-C", dir_path, "status"])

    # If the local repository is behind the remote repository, there are updates available
    if "Your branch is behind" in status.decode("utf-8"):
        print("-> Updates are available")
        subprocess.run(["git", "-C", r"https://github.com/TencentARC/GFPGAN", "pull"])
        print("-> Updates installed")
    else:
        print("-> No updates available")
else:
    print("-> Directory does not exist, cloning the repo...")
    # Clone the repo
    os.system("git clone https://github.com/TencentARC/GFPGAN.git " + dir_path)
    os.system("python ./modelsetup.py develop")
    print("-> repo installed")


#__main__

# Sample input line:
# python main.py --c --l C:\TheGoodShit\StableDiffusion\stable-diffusion-webui\models\Stable-diffusion\beautifulRealistic_v60.safetensors

# Checking for negative prompt efficiancy
basic_neg = "(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime), text, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck"

basic_neg_set = set(basic_neg.split(","))
arg_neg_set= set(args.n.split(","))

# Combine elements, handling potential empty strings
args.n = ",".join(basic_neg_set | arg_neg_set)

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
    elif args.g == True:
        # print(args.size)
        if len(args.p.split(",")) != len(args.n.split(",")):
            valBool = True
        else:
            valBool = False
        pipelineSetup(
            model_path = args.l, 
            prompt = args.p,
            negative_prompt = args.n,
            cfg = args.cfg,
            clip_skip = args.clip_skip,
            steps = args.steps,
            seed = args.s, 
            batch_size = args.batch_size,
            size = args.size,
            use_embeddings = True         
            )
    
    elif args.b == True:
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
                # scale = args.u
                # clip_skip = args.clip_skip,
                # batch_size = args.batch_size,
                # size = args.size  
            )
        else:
            print("[Location Error] Path to/and/or Image do(es) not exist")

    else:
        print("[Generation/Conversion Error] No correct method selected, use '-h' to get list of available methods to use")
else:
    print("[Location Error] Location of model left empty, use --l to add location, use '-h' to get list of available methods to use")