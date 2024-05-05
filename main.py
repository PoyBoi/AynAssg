import os
import argparse
import subprocess

from PIL import Image

from scripts.pipeline_setup import (pipelineSetup, swapBG)

parser = argparse.ArgumentParser()
parser.add_argument('-convert', '--c', '-C', help='Check for if you want to convert a .safetensor model into a diffusor model and store it', action='store_true')
parser.add_argument('-generate', '--g', '-G', help='Sets mode to generate', action='store_true')
parser.add_argument('-background', '--b', '-B', help='Generates the background for an image', action='store_true')
parser.add_argument('-upscale', '--u', '-U', type=int, help='Upscales the image by scale of <x>', default = -1)

parser.add_argument('-setup', '--r', '-R', help='Does a dry run through the code, installing dependancies.', action='store_true')

parser.add_argument('-file', '--f', '-f', help='Pass the location for the image to be used for inpainting', default='')

parser.add_argument('-loc', '--l', '-L', type=str, help='Set the location for the model', default='')
parser.add_argument('-prompt', '--p', '-P', type=str, help='Stores the prompt', default='')
parser.add_argument('-neg-prompt', '--n', '-N', type=str, help='Stores the negative prompt', default='')

parser.add_argument('-seed', '--s', '-S', type=int, help='Seed for generating the image', default = -1)
parser.add_argument('-cfg', type=float, help='How imaginative the AI is, from a scale of 1 to 15', default=7)
parser.add_argument('-clip-skip', type=int, help='Accounts for the CLIP skip setting', default=1)
parser.add_argument('-steps', type=int, help='The amount of inference sampling steps the models takes', default=20)
parser.add_argument('-batch-size', type=int, help='Controls the number of images generated at once', default=1)

parser.add_argument('-size', nargs='+', help='Input the size of the image in W H', default=[512, 512])

parser.add_argument('-lora', type=str, help="Location of lora to be applied, if any", default='')

parser.add_argument('-maskloc', '--k', '-K', type=str, help="Location of mask to be applied, if any", default='')

parser.add_argument('-method', '--z', '-z', type=int, help="Which sampler to be used", default=99)

args = parser.parse_args()

if os.name == 'nt':  # Windows
    os.system('cls')
else:  # Unix-like systems (macOS, Linux)
    os.system('clear')

print("---> Fetching CWD")
current_dir = os.path.dirname(os.path.realpath(__file__))
current_dir = current_dir.split("\\")
current_dir = "/".join(current_dir) + '/'
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


print("---> Checking for GFPGAN repo installation")
dir_path = "models/dependancy/GFPGAN"
if os.path.isdir(dir_path):
    print("-> Already installed")
    print("-> Checking for updates")
    subprocess.run(["git", "-C", dir_path, "fetch"])

    status = subprocess.check_output(["git", "-C", dir_path, "status"])

    if "Your branch is behind" in status.decode("utf-8"):
        print("-> Updates are available")
        subprocess.run(["git", "-C", r"https://github.com/TencentARC/GFPGAN", "pull"])
        print("-> Updates installed")
    else:
        print("-> No updates available")
else:
    print("-> Directory does not exist, cloning the repo...")
    os.system("git clone https://github.com/TencentARC/GFPGAN.git " + dir_path)
    os.system("cd ./models/dependancy/GFPGAN && python setup.py develop")
    print("-> repo installed")

print("\n ---> Checking for GFPGAN's model installation")
if os.path.isfile("./models/dependancy/GFPGAN/experiments/pretrained_models/GFPGANv1.3.pth"):
    print("-> Core model exists, proceeding...\n")
else:
    print("-> Core model does not exist, downloading...\n")
    subprocess.run(["curl", "-LJ", "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth", "-o", r"C:/Users/parvs/VSC Codes/Python-root/AynAssg/models/dependancy/GFPGAN/experiments/pretrained_models/GFPGANv1.3.pth"])
    print("-> Model downloaded @ AynAssg/models/dependancy/GFPGAN/experiments/pretrained_models/ \n") 


#__main__

# Sample input line:
# python main.py --c --l C:\TheGoodShit\StableDiffusion\stable-diffusion-webui\models\Stable-diffusion\beautifulRealistic_v60.safetensors

# Checking for negative prompt efficiancy
basic_neg = "(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime), text, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck"

basic_neg_set = set(basic_neg.split(","))
arg_neg_set= set(args.n.split(","))

# Combine elements, handling empty strings
args.n = ",".join(basic_neg_set | arg_neg_set)

if args.u != -1 or args.r == True:
    args.l = "placeholder"

if args.l !='':
    loc = args.l
    file_name = loc.split("\\")[-1].split(".")[0]
    
    # Removing incorrect literals
    noUse = ['-', '_', '.', '--', '..']
    for i in file_name:
        if i in noUse:
            file_name_new = file_name.split(i)[0]
            file_name = file_name_new
            break

    print("------> Look for the model inside: models\diffused\{}\n".format(file_name))

    # Model conversion pipeline
    if args.c == True:
        print("Converting {} to diffusor based model".format(file_name))
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
            use_embeddings = True,
            method= args.z        
            )
        
    elif args.r == True:
        print("Dry run has been completed successfully, depenadancies have been installed")        

    elif args.u != -1:
        print("-> Running Upscaling...\n")
        a = subprocess.run(["python", r"./models/dependancy/GFPGAN/inference_gfpgan.py", "-i", str(args.f), "-o" ,"results","-s" , str(args.u), "--bg_upsampler", "realesrgan"], capture_output=True, text=True)
        print("-> Upscaling done, printing output...\n")
        print(a.stderr)
        print(r"-> Look for output inside AynAssg/results/restored_imgs")

        loc = args.f.split("\\")[-1]
        i = Image.open(r"./results/restored_imgs/{}".format(loc))
        i.show()
    
    elif args.b == True:
        if os.path.isfile(args.f):
            swapBG(
                model_path = args.l, 
                prompt_bg = args.p,
                negative_prompt_bg = args.n,
                filename = args.f,
                cfg = args.cfg,
                steps = args.steps,
                seed = args.s,
                cmask = args.k,
                # scale = args.u
                # clip_skip = args.clip_skip,
                batch_size = args.batch_size,
                method = args.z
                # size = args.size  
            )
        else:
            print("[Location Error] Path to/and/or Image do(es) not exist")
    else:
        print("[Generation/Conversion Error] No correct method selected, use '-h' to get list of available methods to use")
else:
    print("[Location Error] Location of model left empty, use --l to add location, use '-h' to get list of available methods to use")