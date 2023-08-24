import json
import requests
import asyncio
import time
from abc import ABCMeta, abstractmethod
import sys 
sys.path.append('.')


RPM = 1/10
rate_limiter = asyncio.Semaphore(10)


class QAMODEL(metaclass=ABCMeta):
    def __init__(self,model_base_type="",model_name=""):
        self.model_base_type = model_base_type
        self.model_name = model_name

    def __repr__(self):
        return f"{self.model_name}"

    @abstractmethod
    def repair(self, data:dict):
        pass


class OPENAI_MODEL(QAMODEL):
    def __init__(self, config, model_name="gpt-3.5-turbo"):
        super().__init__(model_base_type="openai", model_name=model_name)
        self.config = config
        self.messages = []
        self.OPENAI_API_KEY = {"gpt-3.5-turbo": [], 
                               "gpt-4": []}
        self.OPENAI_REQUEST_HEADER = {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + self.OPENAI_API_KEY[self.model_name][0]["KEY"],
        }
        self.OPENAI_BASE_URL = self.OPENAI_API_KEY[self.model_name][0]["BASE_URL"]

    def get_response(self, messages):             
        # contruct prompt of openai
        request_body = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": 1024,
            "stream": False,
            "n": self.config.candidate_nums,
        }
        try:
            try:
                response = requests.post(self.OPENAI_BASE_URL, json=request_body, headers=self.OPENAI_REQUEST_HEADER, timeout=60)
                response = json.loads(response.text)
                print("response: ", response)
                response = [response["choices"][i]["message"]["content"].strip() for i in range(self.config.candidate_nums)]
            except:
                print(f"request failed, messages: {messages}, try again...")
                time.sleep(10)
                OPENAI_REQUEST_HEADER = self.OPENAI_REQUEST_HEADER
                OPENAI_BASE_URL = self.OPENAI_BASE_URL
                try:
                    response = requests.post(OPENAI_BASE_URL, json=request_body, headers=OPENAI_REQUEST_HEADER, timeout=90)
                    response = json.loads(response.text)
                    print("response: ", response)
                    response = [response["choices"][i]["message"]["content"].strip() for i in range(self.config.candidate_nums)]
                except:
                    time.sleep(10)  
                    response = requests.post(OPENAI_BASE_URL, json=request_body, headers=OPENAI_REQUEST_HEADER, timeout=90)
                    response = json.loads(response.text)
                    print("response: ", response) 
                    response = [response["choices"][i]["message"]["content"].strip() for i in range(self.config.candidate_nums)]           
        except:
            response = None       
        return response
    
    async def repair(self, data):
        async with rate_limiter:
            loop = asyncio.get_event_loop()
            messages = [{"role": "system", "content": self.config.prompt}]
            if "examples" in data:
                for example in data["examples"]:
                    messages += [{"role": "user", "content": "Problem Description: " + example["problem_description"] + "\n\nWrong Code: " + eval(example["codes"])[0]}, {"role": "assistant", "content": "Here is the correct code:\n```python\n" + eval(example["codes"])[-1] + "\n```"}]
            messages += [{"role": "user", "content": "Problem Description: " + data["problem_description"]}]
            fix_code = await loop.run_in_executor(None, self.get_response, messages)
            return fix_code


class ChatGPT(OPENAI_MODEL):
    def __init__(self, config):
        super().__init__(config, model_name="gpt-3.5-turbo")


