# coding: utf-8
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'False'
import warnings
import gc
import time
from typing import Dict

import numpy as np
import torch
import transformers
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig
from transformers import StoppingCriteria, StoppingCriteriaList
from peft import PeftModel, LoraConfig, get_peft_model, prepare_model_for_kbit_training

import spacy
from ltp import LTP
import langdetect
langdetect.DetectorFactory.seed = 0
from utils import get_ents_en, get_ents_zh, add_pinyin, get_labelled_text

import openai
openai.api_key = "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"

# specify tasks
# tasks = ['abs', 'poli', 'trans']
tasks = ['trans']

# specify base model
#base_model = 'bloomz-560m'
base_model = 'bloomz-1b7'
base_model_dir = f'./models/{base_model}'

# specify langauge
lang = 'en'

# specify lora weights
hide_model_path = f"./lora_weights/hide_{base_model}_{lang}/checkpoint-6300"
hide_method = 'model1b7'
#hide_method = 'model560m'
seek_model_path = f"./lora_weights/seek-%s_{hide_method}_{base_model}_{lang}/checkpoint-2700"

# special tokens
DEFAULT_PAD_TOKEN = '[PAD]'
DEFAULT_EOS_TOKEN = '</s>'
DEFAULT_BOS_TOKEN = '<s>'
DEFAULT_UNK_TOKEN = '<unk>'

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=torch.bfloat16,
)

def smart_tokenizer_and_embedding_resize(
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    special_tokens_dict: Dict[str, str] = {}
    if tokenizer.pad_token is None:
        special_tokens_dict['pad_token'] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict['eos_token'] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict['bos_token'] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict['unk_token'] = DEFAULT_UNK_TOKEN
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

def hide_text(raw_input, target_ents, model, tokenizer, lang, ltp, spacy_model):
    sub_model = PeftModel.from_pretrained(model, hide_model_path, quantization_config=bnb_config, device_map='cuda:0', trust_remote_code=True)
    with open(f'./prompts/v5/hide_{lang}.txt', 'r', encoding='utf-8') as f:
        initial_prompt = f.read()
    if target_ents == 'label':
        return get_labelled_text(raw_input, spacy_model, return_ents=False)
    if target_ents == 'auto':
        if lang == 'en':
            target_ents = get_ents_en(raw_input, spacy_model)
        else:
            target_ents = get_ents_zh(raw_input, ltp, spacy_model)
        print(target_ents)
    input_text = initial_prompt % (raw_input, target_ents)
    input_text += tokenizer.bos_token
    inputs = tokenizer(input_text, return_tensors='pt')
    inputs = inputs.to('cuda:0')
    len_prompt = len(inputs['input_ids'][0])
    def custom_stopping_criteria(input_ids: torch.LongTensor, score: torch.FloatTensor, **kwargs) -> bool:
        cur_top1 = tokenizer.decode(input_ids[0,len_prompt:])
        if '\n' in cur_top1 or tokenizer.eos_token in cur_top1:
            return True
        return False
    pred = sub_model.generate(
        **inputs, 
        generation_config = GenerationConfig(
            max_new_tokens = int(len(inputs['input_ids'][0]) * 1.3),
            do_sample=False,
            num_beams=3,
            repetition_penalty=5.0,
            ),
        stopping_criteria = StoppingCriteriaList([custom_stopping_criteria])
        )
    pred = pred.cpu()[0][len(inputs['input_ids'][0]):]
    response = tokenizer.decode(pred, skip_special_tokens=True).split('\n')[0]
    torch.cuda.empty_cache()
    gc.collect()
    return response

def get_api_output(hidden_text, task_type, lang):
    with open(f'./prompts/v5/api_{task_type}_{lang}.txt', 'r', encoding='utf-8') as f:
        template = f.read()
    response = openai.ChatCompletion.create(
            #   model="gpt-4",
              model="gpt-3.5-turbo",
              temperature=0.1,
              messages=[
                    {"role": "user", "content": template % hidden_text}
                ]
            )
    return response['choices'][0]['message']['content'].strip(" \n")

def recover_text(sub_content, sub_output, content, model, tokenizer, task_type, lang):
    re_model = PeftModel.from_pretrained(model, seek_model_path % task_type, quantization_config=bnb_config, device_map='cuda:0', trust_remote_code=True)
    with open(f'./prompts/v5/seek_{task_type}_{lang}.txt', 'r', encoding='utf-8') as f:
        initial_prompt = f.read()
    input_text = initial_prompt % (sub_content, sub_output, content)
    input_text += tokenizer.bos_token
    inputs = tokenizer(input_text, return_tensors='pt')
    inputs = inputs.to('cuda:0')
    len_prompt = len(inputs['input_ids'][0])
    def custom_stopping_criteria(input_ids: torch.LongTensor, score: torch.FloatTensor, **kwargs) -> bool:
        cur_top1 = tokenizer.decode(input_ids[0,len_prompt:])
        if '\n' in cur_top1 or tokenizer.eos_token in cur_top1:
            return True
        return False
    pred = re_model.generate(
        **inputs, 
        generation_config = GenerationConfig(
            max_new_tokens=1024,
            do_sample=False,
            num_beams=3,
            ),
        stopping_criteria = StoppingCriteriaList([custom_stopping_criteria])
        )
    pred = pred.cpu()[0][len(inputs['input_ids'][0]):]
    recovered_text = tokenizer.decode(pred, skip_special_tokens=True).split('\n')[0]
    torch.cuda.empty_cache()
    gc.collect()
    return recovered_text

if __name__ == '__main__':
    # load models
    print('loading model...')
    model = AutoModelForCausalLM.from_pretrained(base_model_dir, load_in_4bit=True, quantization_config=bnb_config, device_map='cuda:0', trust_remote_code=True, torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(base_model_dir, trust_remote_code=True)
    smart_tokenizer_and_embedding_resize(tokenizer=tokenizer,model=model)
    spacy_model = spacy.load(f'{lang}_core_web_trf')
    # only chinese uses ltp
    ltp = LTP("LTP/small")
    if torch.cuda.is_available():
        ltp.cuda()
    while True:
        # input text
        raw_input = input('\033[1;31minput:\033[0m ')
        if raw_input == 'q':
            print('quit')
            break
        # hide
        target_ents = input('\033[1;31mtarget entities:\033[0m ')
        hidden_text = hide_text(raw_input, target_ents, model, tokenizer, lang, ltp, spacy_model)
        print('\033[1;31mhide_text:\033[0m ', hidden_text)
        # seek
        for task_type in tasks:
            sub_output = get_api_output(hidden_text, task_type, lang).replace('\n', ';')
            print(f'\033[1;31mhidden output for {task_type}:\033[0m ', sub_output)
            if lang == 'zh' and task_type == 'trans':
                raw_input = add_pinyin(raw_input, ltp)
            output_text = recover_text(hidden_text, sub_output, raw_input, model, tokenizer, task_type, lang)
            print(f'\033[1;31mrecovered output for {task_type}:\033[0m ', output_text)
