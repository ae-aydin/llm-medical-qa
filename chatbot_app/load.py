from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel 

from dotenv import load_dotenv
import os
import json

import warnings
warnings.filterwarnings("ignore")


class ModelLoader:
    def __init__(self):
        self.strings = self._load_strings()
    
    def _load_token(self):
        load_dotenv()
        hf_token = os.getenv("HF_TOKEN")
        return hf_token

    def _load_strings(self):
        with open("strings.json", "r") as f:
            strings = json.load(f)
        return strings
    
    def load_model(self):
        base_model = AutoModelForCausalLM.from_pretrained(self.strings["bnb_base"], low_cpu_mem_usage=True, device_map="auto", token=self._load_token())
        tokenizer = AutoTokenizer.from_pretrained(self.strings["bnb_base"], token=self._load_token())
        model = PeftModel.from_pretrained(base_model, self.strings["adapter"])
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.model = model