# Model REVOLVER

Model REVOLVER (Rapid Evolution Via Optimized-List Viewer Evaluated Response) is a Python script designed to merge and evaluate two machine learning models using Hugging Face's Transformers. The tool makes `x` cycles of model mergers, reads prompts from a text file, generates responses, and saves relevant data to individual text files. When the cycles are complete, the system notifies the user to review the outputs, allowing them to choose the cycle number that best fits the model they want to assemble.

## Table of Contents

- [Installation](#installation)
- [Utility](#utility)
- [Practical Use Case](#practical-use-case)
- [Usage](#usage)
- [License](#license)

## Installation

### Prerequisites

- Python 3.6 or higher
- CUDA compatible GPU (Optional but recommended)

### Steps

1. **Clone the repository:**

   ```bash
   git clone https://github.com/Digitous/ModelREVOLVER.git
   cd model-revolver
   ```

2. **Install the required libraries:**

   ```bash
   pip install transformers colorama numpy datasets
   ```

3. **Download the models you want to merge and place them in appropriate directories.**

4. **Create a text file named `prompts.txt` containing the prompts for evaluation; Alpaca and Vicuna as well as any multiline instruct or non instruct prompt works as the entire prompts.txt is taken as the target prompt.**

## Utility

Model REVOLVER empowers your model merge using order through chaos; you define the prompt in prompts.txt, then in command line select two models and the number of cycles - cycles are the number of times the script will merge using randomized ratios per layer. Think of it as {model A -> model B} ratios determine how much A is injected into B per-layer. The system will automatically load both models to CPU memory and model B will be modified in memory by system selected ratios. Everything is recorded to text files from cycle number, to each ratio per layer as well as five prompt completions per cycle. The merged model is automatically created, prompts triggered, then the merged model is destroyed (to prevent a crowd of merged models under testing maxing storage space). The goal is the telemetry for which merge ratios produce the most desirable results to the end user. After the number of cycles the user defined is completed - the script will present five completions for the prodivded prompt for every cycle produced. The user is prompted to examine the completions per cycle in the command line, and is further prompted to select the cycle number of the user's favorite five completions. Upon selecting - the system will rebuild the winning model and automatically move necessary files to the directory to have a composite, coherent, and desired model for the user through the power of random space search layer randomization offers.

## Practical Use Case

Model REVOLVER can be used by researchers, data scientists, and developers who want to experiment with different model combinations. By merging various models, users can potentially create more robust and specialized models tailored to specific tasks or domains.

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
python model_revolver.py --firstmodel /path/to/first/model --secondmodel /path/to/second/model --mergedpath /path/to/save/merged/model
```
Highly recommend making a .bat or .sh to make the process easier; system is command line based for semi automation as well as convenience purposes for those who do not have a GUI interface.

### Workflow

1. **Merge Models:** Combines layers of two models based on randomly generated merge ratios.
2. **Generate Responses:** Uses the merged model to generate responses to prompts from `prompts.txt`.
3. **Review Outputs:** After all cycles, users review the outputs and select the best cycle.
4. **Recreate Model:** The selected model is recreated and saved based on the user's choice.

## License

This project is open-source and available under the [MIT License](LICENSE).

---

Feel free to modify the README to include additional details or sections as needed for your project.
