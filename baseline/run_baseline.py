import os
import torch
import pickle
import asyncio
import random
from tqdm import tqdm
from models import ChatGPT, GPT4, Claude, CodeGeex2, StarCoderChat, PaLM, ChatGLMPro


models = {"ChatGPT":ChatGPT, "Claude":Claude, "CodeGeex2":CodeGeex2, "StarCoderChat":StarCoderChat, "PaLM":PaLM, "ChatGLMPro":ChatGLMPro}

class Config():
    def __init__(self, model="CodeGeex2", few_shot=0, prompt="", device="cuda", candidate_nums=5):
        self.model = model
        self.prompt = prompt
        self.device = device
        self.few_shot = few_shot
        self.candidate_nums = candidate_nums


async def main(config):
    use_ckpt = False
    with open("./test.pkl", "rb") as f:
        df = pickle.load(f)
    datasets = df.to_dict(orient='records')
    # load ckpt
    if use_ckpt:
        with open("./ckpt/gpt-4_0_shot_fix_codes_.pkl", "rb") as f:
            ckpt_fix_codes = pickle.load(f)
        todo_idx = [76]
        todo_datasets = [datasets[idx] for idx in todo_idx]
    else:
        todo_datasets = datasets
    if config.few_shot == 0:
        datasets = list(map(lambda x: {"problem_description":x["problem_description"], "wrong_code":eval(x["codes"])[0]}, todo_datasets))
    else:
        datasets = list(map(lambda x: {"problem_description":x["problem_description"], "wrong_code":eval(x["codes"])[0], "examples": random.sample(datasets, config.few_shot)}, todo_datasets))
    print(f"Test dataset nums:{len(datasets)}, data:{datasets[0]}")
    model = models[config.model](config)
    if config.model in ["ChatGPT", "GPT4", "PaLM", "Claude", "ChatGLMPro"]:
        tasks = [model.repair(data) for data in datasets]
        fix_codes = await asyncio.gather(*tasks)
    else:   
        fix_codes = []
        for idx, data in tqdm(enumerate(datasets), desc="Repairing..."): 
            fix_code = model.repair(data)
            fix_codes.append(fix_code)

    # reuse ckpt
    if use_ckpt:
        for idx in todo_idx:
            ckpt_fix_codes[idx] = fix_codes.pop(0)
    else:
        ckpt_fix_codes = fix_codes
    print("Repairing done!")
    # save fix codes
    save_dir = os.path.join("./ckpt/", str(model) + "_" + str(config.few_shot) + "_shot_fix_codes_2.pkl")
    with open(save_dir, "wb") as f:
        pickle.dump(ckpt_fix_codes, f)



if __name__ == "__main__":
    prompt = """You will play the role of a programming expert. 
                Given a problem description and wrong code, please fix the errors in the code and provide the correct code. 
                Note that you need to use markdown format for the code section. 
                Please ensure that the code is executable."""
    # Note that you need to use markdown format for the code section. 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = Config(model="ChatGPT", prompt=prompt, few_shot=0, candidate_nums=5)
    asyncio.run(main(config))
