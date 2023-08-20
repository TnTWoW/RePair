import argparse
import torch
from dialogues import DialogueTemplate, get_dialogue_template, prepare_dialogue, prepare_dialogue_with_description, prepare_dialogue_with_description_multi_step
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          GenerationConfig, set_seed)
from datasets import load_dataset
import os
import json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_id",
        type=str,
        help="Name of model to generate samples with",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        help="Name of dataset",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="The model repo's revision to use",
    )
    parser.add_argument(
        "--system_prompt", type=str, default=None, help="Overrides the dialogue template's system prompt"
    )
    args = parser.parse_args()



    ### Load Dataset from https://huggingface.co/datasets/TnT/Multi_CodeNet4Repair
    from datasets import load_dataset
    raw_datasets = load_dataset("TnT/Multi_CodeNet4Repair")
    test_datasets = raw_datasets['test']

    try:
        dialogue_template = DialogueTemplate.from_pretrained(args.model_id, revision=args.revision)
    except Exception:
        print("No dialogue template found in model repo. Defaulting to the `no_system` template.")
        dialogue_template = get_dialogue_template("fix")

    if args.system_prompt is not None:
        dialogue_template.system = args.system_prompt

    test_datasets = test_datasets.map(prepare_dialogue_with_description_multi_step,
                                      fn_kwargs={"dialogue_template": dialogue_template, "is_train": False})

    print("=== SAMPLE PROMPT ===")
    print(test_datasets[0]['text'])
    print("=====================")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, revision=args.revision)
    print(f"Special tokens: {tokenizer.special_tokens_map}")
    print(f"EOS token ID for generation: {tokenizer.convert_tokens_to_ids(dialogue_template.end_token)}")
    generation_config = GenerationConfig(
        temperature=0.2,
        top_k=50,
        top_p=0.95,
        repetition_penalty=1.2,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.convert_tokens_to_ids(dialogue_template.end_token),
        min_new_tokens=32,
        max_new_tokens=512,
        num_return_sequences=5,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, revision=args.revision, device_map="auto", torch_dtype=torch.float16
    )
    outputs = {}
    for idx, prompt in enumerate(test_datasets['text']):
        batch = tokenizer(prompt, return_tensors="pt", return_token_type_ids=False).to(device)
        generated_ids = model.generate(**batch, generation_config=generation_config)
        # generated_texts = tokenizer.decode(generated_ids[0], skip_special_tokens=False).lstrip()
        generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)
        generated_texts = [text.lstrip().replace(tokenizer.eos_token, "") for text in generated_texts]
        outputs[idx] = generated_texts
        print(f"=== EXAMPLE {idx} ===")
        print()
        print(generated_texts[0])
        print()
        print("======================")
        print()


if __name__ == "__main__":
    main()
