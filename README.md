# Model REVOLVER

Model REVOLVER (Rapid Evolution Via Optimized-List Viewer Evaluated Response) is a Python script designed to rapidly merge a user selected pair of models on a semi randomized basis and give the end user the power to evaluate the desired results and pick from a pool of potentials. The tool makes `x` cycles of model merges, reads prompts from a text file, generates responses, and saves relevant data to individual text files. When the cycles are complete, the system notifies the user to review the outputs, allowing them to choose the cycle number that best fits the model they want to assemble, which the system completes automatically for the user's convenience.

## Table of Contents

- [Installation](#installation)
- [Utility](#utility)
- [Practical Use Case](#practical-use-case)
- [Usage](#usage)
- [License](#license)

## Installation

### Prerequisites

- Python 3.6 or higher
- Latest version of HuggingFace's Transformers library (necessary to support on-the-spot 4 bit loading to GPU for assessments)
- CUDA compatible GPU (Mandatory at the moment; roadmap will address this)

### Steps

1. **Clone the repository:**

   ```bash
   git clone https://github.com/Digitous/ModelREVOLVER.git
   cd ModelREVOLVER
   ```

2. **Install the required libraries:**

   ```bash
   pip install transformers colorama numpy datasets
   ```

3. **Download the models you want to merge and place them in appropriate directories. (Must be same architecture and number of parameters - yes, llamav2 is supported)**

4. **Create a text file named `prompts.txt` containing the prompt for evaluation; Alpaca and Vicuna as well as any multiline instruct or non instruct prompt works as the entire prompts.txt is taken as the target prompt.**

## Utility

Model REVOLVER empowers your model merge using order through chaos; you define the prompt in prompts.txt, then in the command line select two models and the number of cycles - cycles are the number of times the script will merge using randomized ratios per layer. Think of it as {model A -> model B} ratios determine how much A is injected into B per-layer. The system will automatically load both models to CPU memory and model B will be modified (by model A) in memory by system selected layer merge ratios. Everything is recorded to text files from cycle number, to each ratio per layer, as well as five prompt completions per cycle based on the prompt provided. Every cycle the merged model is automatically created, prompts triggered, then the merged model is destroyed (to prevent a crowd of merged models under testing maxing storage space). The goal is the telemetry for which merge ratios produce the most desirable results to the end user. After the number of cycles the user defined is completed - the script will present five completions for the prodivded prompt for every cycle produced. The user is prompted to examine the completions per cycle in the command line, and is further prompted to select the cycle number of the most preferable five completions. Upon selecting - the system will rebuild the winning model and automatically move necessary files to the directory and the user will have a composite, coherent, and desired model through the power of random space search layer randomization. Yay.

## Practical Use Case

Model REVOLVER can be used by researchers, data scientists, and developers who want to experiment with different model combinations. By merging various models, users can potentially create more robust and specialized models tailored to specific tasks or domains. Models created through this merge process can be further merged if desired, and further influenced by prompt response choices again. Or simply used as wanted. Through this method, censorship can be influenced to be increased, decreased, verbosity and overall tone altered. Keep in mind it's only as good as the parent models chosen.

## Usage

### Command Line Arguments

- `--firstmodel`: Path to the first model.
- `--secondmodel`: Path to the second model.
- `--mergedpath`: Path to save the merged model.
- `--num_cycles`: Number of merge-eval-delete cycles to perform (default 3).
- `--fp16`: Save the merged model in half precision (fp16) (default False).
- `--shardsize`: Max shard size for the merged model (default "18000MiB").

### Example Command

```bash
python model_revolver.py --firstmodel /path/to/first/model --secondmodel /path/to/second/model --mergedpath /path/to/save/merged/model --cycles 10
```
To keep things simple, it is highly recommend to make a .bat or .sh to make the process easier; system is command line based for semi automation as well as convenience purposes for those who do not have a GUI interface. Full folder path when selecting models in command line is recommended.

### Workflow

1. **Merge Models:** Combines layers of two models based on randomly generated merge ratios.
2. **Generate Responses:** Uses the merged model to generate five responses to prompt from `prompts.txt`.
3. **Review Outputs:** After all cycles, users review the outputs and select the best cycle.
4. **Recreate Model:** The selected model is recreated and saved based on the user's choice.

## License

I don't know anything about licenses. Do whatever you want with it, choose to be cool and give me credit, or don't. I don't really care. Enjoy mixtuning. -Digitous

## Known issues and QoL to address

This script is CPU memory hungry and ironically not VRAM hungry. This is because it loads both selected models in fp32 simultaneously when merge operations are being committed - this is on purpose for the sake of merging at the highest accuracy. On Windows, it should work even with a lower tier of RAM since it will dig into page file if it needs to - but prepare to go make a sandwich, watch TV, complain the damn thing is taking too long. After a cycle has made its modifications to model2 in memory, it will temporarily save it, copy files to the temp folder, and reload it to VRAM in 4 bit for a quick series of inferences based on the prompt. Every cycle will be like this. For reference, I have a 128GB DDR5 RAM and an RTX3090 24GB; however you will not need nearly this level of specs to use this tool, just a bit more patience than me. Thanks to 4 bit loading the amount of VRAM necessary is fairly low, even for a 13B model.

## Update Roadmap

Re-implement non-llama model support (was intended for init release, will be patched soon)
Decrease CPU RAM footprint
Bring several options to the surface through commandline flags
Vague potential to have a lightweight LLM auto-judge outputs based on what the user defines they are looking for in the results (ideally I want this tool to be 100% automatable)

## Friendly Message

Feel free to contact me with insights and feedback. After all, I'm just an idiot with a keyboard who doesn't know when to quit. My ear is open to ideas.
I run CalderaAI on HuggingFace and release models under my name as well as under Caldera. Also check out the hard work of KoboldAI on HuggingFace - if you visit their discord they are the friendliest group of Kobolds one could ever hope to meet.
