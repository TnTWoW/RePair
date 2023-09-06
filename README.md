## Automated Program Repair with Process-based Feedback

This is the official code for the paper Automated Program Repair with Process-based Feedback

### Contents:
- [Overview](#overview)
- [Installation](#installation)
- [Datasets](#datasets)
- [Usage](#usage)

## Overview

* Humans make multiple revisions when writing programs, while LLMs cannot receive feedback from compiler and test cases to optimize its repair policies automatically. We are exploring how to ensure small-scale LLM still outperform through process supervision and feedback.

* During training, we develop a reward model that serves as a critic, providing feedback for the fine-tuned LM’s action, progressively optimizing its policy.

* During inference, we require the LM to generate solutions iteratively until the repair effect no longer improves or hits the maximum step limit.

## Installation

The code requires some dependencies as specified in `requirements.txt`. Please follow the relevant libraries to install or run: 

`pip install -r requirements.txt`

## Datasets

We establish a multi-step program repair dataset based on CodeNet, which we call CodeNet4Repair. It include comprehensive test cases, problem descriptions, and detailed repair steps.

The dataset are available here:

```
https://huggingface.co/datasets/TnT/Multi_CodeNet4Repair
```

## Usage

```python
python inference.py --model_id TnT/process-based-repair --dataset_name TnT/Multi_CodeNet4Repair
```

## GPT-3.5 Prompt
`
You will play the role of a programming expert. 
Given a problem and incorrect code, please fix the errors in the code and provide the correct code. 
Note that you need to use markdown format for the code section. 
Please ensure that the code is executable.
`

## Our model's Prompt
`
Below is a description and wrong answer for the programming problem. Write the correct solution to fix it.
`###` Problem:
