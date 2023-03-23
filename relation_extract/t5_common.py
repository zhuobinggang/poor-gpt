from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import math
import numpy as np
import time
import types
import itertools

def create_model(learning_rate = 2e-5, size = 'small', max_token_size = 512, t5 = None):
    res = types.SimpleNamespace()
    # NOTE
    # res.t5 = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")
    # res.toker = T5Tokenizer.from_pretrained("google/flan-t5-small")
    if t5 is None:
        res.t5 = T5ForConditionalGeneration.from_pretrained(f"google/flan-t5-{size}")
    else:
        res.t5 = t5
    res.max_token_size = max_token_size
    res.size = size
    res.toker = T5Tokenizer.from_pretrained(f"google/flan-t5-{size}", truncation_side= 'left', model_max_length = max_token_size)
    # res.t5 = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")
    # res.toker = T5Tokenizer.from_pretrained("google/flan-t5-base")
    res.opter = torch.optim.AdamW(res.t5.parameters(), learning_rate)
    res.t5.cuda()
    res.t5.train()
    return res

def save_model(m, name):
    torch.save(m.t5, f'./check_points/{name}_size{m.size}_maxtoken{m.max_token_size}.tch')

def load_model(name, size, max_token_size, learning_rate = 2e-5):
    t5 = torch.load(f'./check_points/{name}_size{size}_maxtoken{max_token_size}.tch')
    res = create_model(learning_rate, size, max_token_size, t5)
    return res

def QA(m, context, question):
    input_text = f'question: {question} context: {context}'
    input_ids = m.toker(input_text, return_tensors="pt", truncation = True).input_ids.cuda()
    step_output_text = m.toker.decode(m.t5.generate(input_ids)[0], skip_special_tokens=True)
    return step_output_text



