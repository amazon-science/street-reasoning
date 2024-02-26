# STREET: a Multi-Task Structured Reasoning and Explanation Benchmark

<!-- ![STREET data example](imgs/gsm8k_n_ar_lsat_examples.png) -->

<img src="imgs/gsm8k_n_ar_lsat_examples.png"  width="900">

This repository contains the data and evaluation code for the paper [STREET: a Multi-Task Structured Reasoning and Explanation Benchmark](https://openreview.net/forum?id=1C_kSW1-k0) (ICLR 2023).


# Setting Up Environment

Requirements:

* Python 3
* networkx
* tqdm
* bleurt

Download the bleurt-base-128 model from https://github.com/google-research/bleurt/blob/master/checkpoints.md

# Running Generation Script

This repository contains the annotations and script to generate the data for the STREET dataset.

Execute the following python script:

```
python run_street_data_generation.py
```

The script will download some of the original datasets merge them with our reasoning and explanation annotations.

The final data will be contained within the JSONL files with format `./data/$STREET_TASK$/reasoning_annotated_$SPLIT$.jsonl`.

# Running Evaluation

To run the evaluation code you need to specify the predicted file and bleurt check-point locations. 

## Example

```
python run_evaluation.py \
--pred_file "all_results.jsonl" \
--bleurt_checkpoint "bleurt-base-128" \
--task "scone"
```

For details on the expected arguments you can run:

```
python run_evaluation.py --help
```

## Prediction File Format

The prediction file should be in JSONL format. The expected fields are question ID, the answer, and linearized reasoning graph, as shown below:

```
{
    "id": "GSM8K_0_0c3da8c7d5", 
    "answer": "2", 
    "linearized_output": "sent1 & sent2 -> int1: Janet eats 16 - 3 = 12 eggs per day.; ..."
}
...
```

**NOTE:** The linearized format uses sentX (for premises) and intX (for conclusion) instead of (X) to simplify parsing and scoring.

# Citation

```
@inproceedings{
    ribeiro2023street,
    title={{STREET}: A {MULTI}-{TASK} {STRUCTURED} {REASONING} {AND} {EXPLANATION} {BENCHMARK}},
    author={Danilo Neves Ribeiro and Shen Wang and Xiaofei Ma and Henghui Zhu and Rui Dong and Deguang Kong and Juliette Burger and Anjelica Ramos and zhiheng huang and William Yang Wang and George Karypis and Bing Xiang and Dan Roth},
    booktitle={International Conference on Learning Representations},
    year={2023},
    url={https://openreview.net/forum?id=1C_kSW1-k0}
}
```