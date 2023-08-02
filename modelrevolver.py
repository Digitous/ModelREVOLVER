import sys
import os
import random
import shutil
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import subprocess
from colorama import init, Fore, Style
import numpy as np
from datasets import load_dataset
from math import exp
import random
import flask
import requests
import itertools
import gc

# [currently working on script variant that assembles models based on user choice of best prompt responses; makes x cycles of models, reads prompt from text file, generates five responses, saves cycle num, percentages, and responses to individual text files, then when the cycles complete, it plays a tone so the user knows the system is ready for them to review outputs... when they reviewed all x cycles of 5 prompts each they choose the cycle number that has what they're looking for in the model they want assembled]

# Set default values for optional arguments
DEFAULT_FP16 = False
DEFAULT_MAX_SHARD_SIZE = "18000MiB"
DEFAULT_NUM_CYCLES = 3

# Set up the command-line argument parser
parser = argparse.ArgumentParser(description='Merge two models.')
parser.add_argument('--firstmodel', type=str, required=True, help='Path to the first model.')
parser.add_argument('--secondmodel', type=str, required=True, help='Path to the second model.')
parser.add_argument('--mergedpath', type=str, required=True, help='Path to save the merged model.')
parser.add_argument('--num_cycles', type=int, default=DEFAULT_NUM_CYCLES, help='Number of merge-eval-delete cycles to perform.')
parser.add_argument('--fp16', action='store_true', default=DEFAULT_FP16, help='Save the merged model in half precision (fp16).')
parser.add_argument('--shardsize', type=str, default=DEFAULT_MAX_SHARD_SIZE, help='Max shard size for the merged model.')

args = parser.parse_args()

# Fetch values from command-line arguments
first_model_path = os.path.abspath(args.firstmodel)
second_model_path = os.path.abspath(args.secondmodel)
merged_model_path = os.path.abspath(args.mergedpath)
num_cycles = args.num_cycles + 1
fp16 = args.fp16
max_shard_size = args.shardsize

# Initialize settings
always_output_fp16 = False
verbose_info = True
force_cpu = True

newline = '\n'
def clear_console():
    if os.name == "nt":  # For Windows
        subprocess.call("cls", shell=True)
    else:  # For Linux and macOS
        subprocess.call("clear", shell=True)

with torch.no_grad():
    if fp16:
        torch.set_default_dtype(torch.float16)
    else:
        torch.set_default_dtype(torch.float32)

    device = torch.device('cuda')
    print(device)

    # Ensure output directory exists
    os.makedirs(merged_model_path, exist_ok=True)
    model_nameX = os.path.basename(first_model_path)
    model_nameY= os.path.basename(second_model_path)
    model_nameZ= os.path.basename(merged_model_path)

clear_console()
print(f"Initial Setup...")

def recreate_model(best_cycle, first_model):
    # Read the merge ratios from the corresponding text file
    with open(os.path.join(os.getcwd(), f"{model_nameZ}_mergecycle{best_cycle}.txt"), "r", encoding='utf-8') as file:
        lines = file.readlines()
    # Parse the merge ratios
    merge_ratios = list(map(float, lines[3].strip().strip('[]').split(',')))
    # Load the second model
    print(f"\nLoading Transient Recipient Parent Model {model_nameY} For Recreation of Model {model_nameZ} from cycle {best_cycle} to RAM...")
    second_model = AutoModelForCausalLM.from_pretrained(second_model_path).to('cpu')
    second_model.eval()
    print("Recipient Loaded. Dtype: " + str(second_model.dtype))
    num_layers = first_model.config.num_hidden_layers
    print("Number of Layers:", num_layers)
    print("Merge Ratios:", merge_ratios)
    # Merge the models according to the stored merge ratios
    for i in range(num_layers):
        first_ratio = merge_ratios[i]
        second_ratio = 1 - first_ratio
        merged_layer = (first_model.model.layers[i].state_dict(), second_model.model.layers[i].state_dict())
        for key in merged_layer[0].keys():
            merged_layer[0][key] = first_ratio * merged_layer[0][key] + second_ratio * merged_layer[1][key]
        second_model.model.layers[i].load_state_dict(merged_layer[0])
        print("Merging Layer " + str(i))
    # Save the merged model
    print(f"{Fore.YELLOW}\nSaving User Preference Cycle {best_cycle} to disk and copying files.{Style.RESET_ALL}")
    second_model.save_pretrained(merged_model_path, max_shard_size=max_shard_size)
    # List of files to copy to merged model dir
    files_to_copy = ["special_tokens_map.json", "tokenizer_config.json", "vocab.json", "tokenizer.model", "generation_config.json", "added_tokens.json", "merges.txt"]
    # Check for the existence of 'special_tokens_map.json' in both directories
    first_model_has_special_tokens = os.path.exists(os.path.join(first_model_path, "special_tokens_map.json"))
    second_model_has_special_tokens = os.path.exists(os.path.join(second_model_path, "special_tokens_map.json"))
    # Decide the source directory based on the presence of 'special_tokens_map.json'
    if first_model_has_special_tokens and not second_model_has_special_tokens:
        src_dir = first_model_path
    elif second_model_has_special_tokens or not first_model_has_special_tokens:
        src_dir = second_model_path
    # Copy each file to the new folder
    for filename in files_to_copy:
        src_path = os.path.join(src_dir, filename)
        dst_path = os.path.join(merged_model_path, filename)
        print(f"\nCopying files from dir: {src_path}")
        print(f"To dir: {dst_path}")
        try:
            shutil.copy2(src_path, dst_path)
        except FileNotFoundError:
            print("\nFile " + filename + " not found in " + src_dir + ". Skipping (likely not important).")  
    del second_model


def review_files(first_model):
    files = sorted([f for f in os.listdir(os.getcwd()) if f.startswith(f'{model_nameZ}_mergecycle') and f.endswith('.txt')])
    for file in files:
        # Extract cycle number from filename
        cycle_num = int(file.replace(f'{model_nameZ}_mergecycle', '').replace('.txt', ''))
        print(f"\nCycle Number: {cycle_num}\n")
        with open(file, "r") as f:
            print(f.read())
        print("\n")
    
    best_cycle = int(input("Please enter the cycle number with the best responses: "))
    # Logic for recreating the model based on the user's choice
    recreate_model(best_cycle, first_model)

# loads from file called prompts.txt, uses that as the prompt and has as many completions as seed is left per this while loop.
def generate_responses(second_model, tokenizer, cycle):
    with open('prompts.txt', 'r', encoding='utf-8') as file:
        prompt = file.read()
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        input_ids = input_ids.to('cuda')
        gen_tokens = second_model.generate(input_ids, max_length=300, num_return_sequences=5, do_sample=True)
        model_completion = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
        with open(os.path.join(os.getcwd(), f'{model_nameZ}_mergecycle{cycle + 1}.txt'), "a", encoding='utf-8') as file:
            print(f'Prompt: {prompt}')
            file.write(f'Prompt: {prompt}')
            for i, completion in enumerate(model_completion):
                print(f'Completion {i+1}: {completion}\n')
                file.write(f'Completion {i+1}: {completion}\n')

def main():
    # Check if paths exist
    if not os.path.exists(first_model_path) or not os.path.exists(second_model_path):
        print("\nYou must select two directories containing models to merge and one output directory. Exiting.")
        exit()
    clear_console()
    print(f"{Fore.YELLOW}[Model REVOLVER: Rapid Evolution Via Optimized-List Viewer Evaluated Response] is working with\nmodels: {Fore.GREEN}{model_nameX}{Fore.YELLOW} and {Fore.GREEN}{model_nameY}{Fore.YELLOW} for {Fore.GREEN}{num_cycles}{Fore.YELLOW} cycles.{Style.RESET_ALL}\n")
    sys.setrecursionlimit(2000)
    # Load the first model before the cycles begin, so it never gets reloaded.
    print(f"Loading Resident Donor Parent Model {model_nameX} to RAM...")
    first_model = AutoModelForCausalLM.from_pretrained(first_model_path).to('cpu')
    #first_model = first_model.to(device)    
    first_model.eval()
    print("Donor Loaded. Dtype: " + str(first_model.dtype))

    # ------------------------cycles begin------------------------
    for cycle in range(num_cycles):
        if cycle == num_cycles - 1:  # If it's the last cycle
            print(f"Initiating Human Review Process...")
            review_files(first_model)
        else:
            random.seed()
            print(f"\nLoading Transient Recipient Parent Model {model_nameY} to RAM for cycle {cycle + 1} of {num_cycles}...")
            second_model = AutoModelForCausalLM.from_pretrained(second_model_path).to('cpu')
            second_model.eval()
            print("Recipient Loaded. Dtype: " + str(second_model.dtype))

            num_layers = first_model.config.num_hidden_layers
            print("Number of Layers:", num_layers)

            merge_ratios = [round(random.uniform(0.0, 1.0), 2) for _ in range(num_layers)]
            print("Merge Ratios:", merge_ratios)

            with open(os.path.join(os.getcwd(), f'{model_nameZ}_mergecycle{cycle + 1}.txt'), "w", encoding='utf-8') as file:
                file.write(f"First Model: {first_model_path}\n")
                file.write(f"Second Model: {second_model_path}\n")
                file.write("Merge Ratios:\n")
                file.write(f"{merge_ratios}\n")

            for i in range(num_layers):
                first_ratio = merge_ratios[i]
                second_ratio = 1 - first_ratio
                merged_layer = (first_model.model.layers[i].state_dict(), second_model.model.layers[i].state_dict())
                for key in merged_layer[0].keys():
                    merged_layer[0][key] = first_ratio * merged_layer[0][key] + second_ratio * merged_layer[1][key]
                second_model.model.layers[i].load_state_dict(merged_layer[0])
                print("Merging Layer " + str(i))

            print(f"{Fore.YELLOW}\nSaving Offspring {cycle + 1} to disk and copying files.{Style.RESET_ALL}")
            second_model.save_pretrained(merged_model_path, max_shard_size=max_shard_size)
            # List of files to copy to merged model dir
            files_to_copy = ["special_tokens_map.json", "tokenizer_config.json", "vocab.json", "tokenizer.model", "generation_config.json", "added_tokens.json", "merges.txt"]
            
            # Check for the existence of 'special_tokens_map.json' in both directories
            first_model_has_special_tokens = os.path.exists(os.path.join(first_model_path, "special_tokens_map.json"))
            second_model_has_special_tokens = os.path.exists(os.path.join(second_model_path, "special_tokens_map.json"))
            
            # Decide the source directory based on the presence of 'special_tokens_map.json'
            if first_model_has_special_tokens and not second_model_has_special_tokens:
                src_dir = first_model_path
            elif second_model_has_special_tokens or not first_model_has_special_tokens:
                src_dir = second_model_path
            
            # Copy each file to the new folder
            for filename in files_to_copy:
                src_path = os.path.join(src_dir, filename)
                dst_path = os.path.join(merged_model_path, filename)
                print(f"\nCopying files from dir: {src_path}")
                print(f"To dir: {dst_path}")
                try:
                    shutil.copy2(src_path, dst_path)
                except FileNotFoundError:
                    print("\nFile " + filename + " not found in " + src_dir + ". Skipping (likely not important).")  
            
            del second_model

            print(f"{Fore.YELLOW}\nLoading Offspring {cycle + 1} to VRAM in 4bit mode for evaluation.{Style.RESET_ALL}")
            
            # loads second_model to GPU in 4 bit for inference (this method was introduced to HuggingFace after September 2022, functions exactly the same as a regular model after it is loaded)
            gc.collect()
            torch.cuda.empty_cache()
            second_model = AutoModelForCausalLM.from_pretrained(merged_model_path, device_map="auto", load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
            tokenizer = AutoTokenizer.from_pretrained(second_model_path)

            print(f"{Fore.YELLOW}\nInitiating Prompt Responses...{Style.RESET_ALL}")
            generate_responses(second_model, tokenizer, cycle)

            del second_model
            gc.collect()
            torch.cuda.empty_cache()
            shutil.rmtree(merged_model_path)
            clear_console()
            
if __name__ == "__main__":
    main()
