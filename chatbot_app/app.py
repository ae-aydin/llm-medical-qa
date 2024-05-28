import gradio as gr
from transformers import TextIteratorStreamer
from threading import Thread
from load import ModelLoader

import warnings
warnings.filterwarnings("ignore")

model_loader = ModelLoader()
model_loader.load_model()


def chat(
    message: str,
    history: list,
    temperature: float,
    max_new_tokens: int,
    system_prompt: str
):
    
    model = model_loader.model
    tokenizer = model_loader.tokenizer
    
    conversation = [{"role": "system", "content": system_prompt}, {"role": "user", "content": message}]

    input_ids = tokenizer.apply_chat_template(
        conversation, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>"),
    ]

    streamer = TextIteratorStreamer(
        tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True
    )

    generate_kwargs = dict(
        input_ids=input_ids,
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=terminators,
        do_sample=True,
        top_p=0.95,
        top_k=1000,
        temperature=temperature,
    )

    if temperature == 0:
        generate_kwargs["do_sample"] = False

    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()

    outputs = []
    for text in streamer:
        outputs.append(text)
        yield "".join(outputs)

chatbot = gr.Chatbot(height=750, label="Medical Chatbot")
textbox = gr.Textbox(placeholder="Enter message here", container=False, scale=7)

with gr.Blocks(fill_height=True) as demo:
    gr.ChatInterface(
        fn=chat,
        chatbot=chatbot,
        textbox=textbox,
        theme="soft",
        additional_inputs_accordion=gr.Accordion(
            label="Parameters", open=False, render=False
        ),
        additional_inputs=[
            gr.Slider(
                minimum=0,
                maximum=1,
                step=0.05,
                value=0.0,
                label="Temperature",
                render=False,
            ),
            gr.Slider(
                minimum=128,
                maximum=2048,
                step=128,
                value=256,
                label="Max New Tokens",
                render=False,
            ),
            gr.Textbox(value=model_loader.strings["system_prompt"], label="System Prompt", render=False)
        ],
        examples=[[example] for example in model_loader.strings["examples"]],
        cache_examples=False,
    )

if __name__ == "__main__":
    demo.launch()
