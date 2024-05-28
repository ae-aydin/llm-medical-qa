# LLM Fine-tuning - Medical QA

**AIN413 Project**

Base Quantized Model: https://huggingface.co/ae-aydin/LLama-3-8B-Instruct-Medical-BNB-Base

LoRA Finetune Model: https://huggingface.co/ae-aydin/Llama-3-8B-Instruct-Medical-QLoRA

How to use:

```
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM

base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct-Medical-BNB-Base")
model = PeftModel.from_pretrained(base_model, "ae-aydin/Llama-3-8B-Instruct-Medical-QLoRA")
```
