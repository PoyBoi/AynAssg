import os
import argparse
import subprocess



parser = argparse.ArgumentParser()
parser.add_argument('-convert', '--c', '-C', help='Check for if you want to convert a .safetensor model into a diffusor model and store it', action="store_true")
parser.add_argument('-generate', '--g', '-G', help='Sets mode to generate', action='store_true')
parser.add_argument('-loc', '--l', '-L', type=str, help='Set the location for the model')
parser.add_argument('-lora', type=str, help="Location of lora to be applied, if any")
parser.add_argument('-prompt', '--p', '-P', type=str, help='Stores the prompt')
parser.add_argument('-seed', '--s', '-S', type=int, help='Seed for generating the image', default=-1)
parser.add_argument('-cfg', type=int, help='How imaginate the AI is, from a scale of 1 to ', default=7)

args = parser.parse_args()



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



if args.l !='':
    loc = args.l
    file_name = loc.split("\\")[-1].split(".")[0]
    if args.c == True:
        print("Converting {} to diffusor based model".format(file_name))
        os.makedirs("models\diffused\{}".format(file_name), exist_ok=True)
        command = "python models\convertModel.py  --checkpoint_path {} --dump_path \models\diffused\{}\ --from_safetensors".format(loc, file_name)
        process = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
        output, error = process.communicate()
        print("Diffusor stored at \models\diffused\{}".format(file_name))