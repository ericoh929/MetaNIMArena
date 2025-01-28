import json
import openai
import os
import random
from collections import Counter
import argparse
import json
import re
from openai import OpenAI
import torch
import transformers
import time
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from pathlib import Path
from tqdm import tqdm
import os
import openai
from sklearn.metrics.pairwise import cosine_similarity

os.environ['CUDA_VISIBLE_DEVICES'] = '4, 5'

import google.generativeai as genai

openai.api_key = os.getenv("OPENAI_API_KEY")

from openai import OpenAI
client = OpenAI()


parser = argparse.ArgumentParser(description='arguments for training')

parser.add_argument('--agent_model',     type=str,   default=None, help='model')
parser.add_argument('--num_games',     type=int,   default='50', help='prompt_method')
parser.add_argument('--look_ahead',     type=int,   default='0', help='prompt_method')

args = parser.parse_args()

def _build_model():
    if args.agent_model == 'llama':
        base_model = '/home/jihwan/LLM/local_model/llama3.1_8b/instruct'
    elif args.agent_model == 'gemma':
        base_model = 'google/gemma-2-9b-it'
    elif args.agent_model == 'qwen':
        base_model = 'Qwen/Qwen2.5-7B-Instruct'
    elif args.agent_model == 'mistral':
        base_model = 'mistralai/Mistral-7B-Instruct-v0.3'
    
    model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map = 'auto'
        )

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    return model, tokenizer

if args.agent_model == 'llama' or args.agent_model == 'gemma' or args.agent_model == 'qwen' or args.agent_model == 'mistral':
    print("Small Language Models: ", args.agent_model)
    model, tokenizer = _build_model()

def convert_to_llama_format(system, instruction): #for instruct-fine-tuned model
    alpaca_format_str = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

    {system} <|eot_id|>
    <|start_header_id|>user<|end_header_id|>

    {instruction}<|eot_id|>
    <|start_header_id|>assistant<|end_header_id|>
    """
    return alpaca_format_str

def say_model(system, instruction_str, model=model, tokenizer=tokenizer):
    if args.agent_model == 'llama':
        inputs = tokenizer(convert_to_llama_format(system, instruction_str), return_tensors = "pt").to("cuda")
        outputs = model.generate(**inputs, max_new_tokens = 500, use_cache = True, temperature = 0.7, top_p = 0.95, pad_token_id = tokenizer.eos_token_id)
        return(tokenizer.batch_decode(outputs)[0])
    
    elif args.agent_model == 'gemma':
        # <bos><start_of_turn>user
        # Write a hello world program<end_of_turn>
        # <start_of_turn>model
        messages = [
        {"role": "user", "content": f"{system}\n{instruction_str}"},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to("cuda")
        outputs = model.generate(input_ids=inputs, max_new_tokens=500,)
        return(tokenizer.batch_decode(outputs)[0])

    elif args.agent_model == 'qwen':
        messages = [
        {"role": "system", "content": f"{system}"},
        {"role": "user", "content": f"{instruction_str}"}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
        generated_ids = model.generate(**inputs, max_new_tokens=500, temperature = 0.7, top_p = 0.95,)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)]
        return(tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0])
    
    elif args.agent_model == 'mistral':
        messages = [{"role": "user", "content": f"{system}\n{instruction_str}"},]
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_dict=True, return_tensors="pt")
        prompt.to("cuda")
        outputs = model.generate(**prompt, max_new_tokens=500, temperature = 0.7, top_p = 0.95,)
        # inputs = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to("cuda")
        return(tokenizer.batch_decode(outputs[0], skip_special_tokens=True))

def get_agent_response(agent, prompt, system_prompt="You are a rational and smart assistant."):
    """
    Handles responses for different models based on the agent's configuration.

    Parameters:
        agent (dict): Dictionary containing agent details, including the model name.
        prompt (str): The input prompt for the model.
        system_prompt (str): The system-level instruction for the model.

    Returns:
        dict: Parsed content from the model's response in JSON format, or None if parsing fails.
    """
    def parse_content(response, split_key):
        try:
            if split_key not in response:
                return None
            content = response.split(split_key)[1]
            matches_with_braces = re.search(r'\{.*?\}', content, re.DOTALL)
            parsed_content_with_braces = matches_with_braces.group(0).replace('\xa0', '').strip() if matches_with_braces else None
            return json.loads(parsed_content_with_braces) if parsed_content_with_braces else None
        except (AttributeError, json.JSONDecodeError, IndexError):
            return None

    while True:
        try:
            if args.agent_model == "llama":
                split_key = "<|start_header_id|>assistant<|end_header_id|>\n"
                response = say_model(system_prompt, prompt)
                # print('llama response: ', response)
                parsed_content = parse_content(response, split_key)
                # print('llama parsed content: ', parsed_content)
                if parsed_content:
                    return parsed_content

            elif args.agent_model == "gemma":
                split_key = "<start_of_turn>model\n"
                response = say_model(system_prompt, prompt)
                parsed_content = parse_content(response, split_key)
                if parsed_content:
                    return parsed_content

            print("Error encountered. Retrying in 2 seconds...")
            time.sleep(2)  # Delay to prevent rapid retries
        except KeyboardInterrupt:
            print("Process interrupted by user.")
            return None
        except Exception as e:
            print(f"Unexpected error: {e}. Retrying...")


def get_embedding(text, model="text-embedding-3-small"):
   text = text.replace("\n", " ")
   return client.embeddings.create(input = [text], model=model).data[0].embedding

def extract_reasoning(prompt, action):
    input_prompt = f"""
    Given the following game scenario:
    {prompt}
    
    The action taken is: {action}.
    
    Your task is to provide the reasoning for this action.

    The output should be a markdown code snippet formatted in the following schema, including the leading and trailing \\`\\`\\`json" and "\\`\\`\\`":\n\n```\n{{\n\t"reasoning": string  // This is the reasons for the action given prompt.\n}}
    """
    # Call the LLM to generate new reasoning
    response = get_agent_response(args.agent_model, input_prompt)
    reasoning = response.get("reasoning")
    return reasoning.strip()

def are_reasonings_equivalent(reasoning1, reasoning2):
    input_prompt = f"""
    Compare the following two reasoning statements and determine if they have the same semantic meaning:
    
    Reasoning 1: "{reasoning1}"\n
    Reasoning 2: "{reasoning2}"
    
    Respond with "Yes" if they have the same semantic meaning, and "No" if they do not.

    The output should be a markdown code snippet formatted in the following schema, including the leading and trailing \\`\\`\\`json" and "\\`\\`\\`":\n\n```\n{{\n\t"reasoning": string  // provide a brief explanation for your response to ensure clarity.\n\t"answer": "Yes" or "No"  // This is an answer to to instruction. Only provide "Yes" or "No"\n}}
    """
    # Call the LLM to evaluate equivalence
    response = get_agent_response(args.agent_model, input_prompt)
    answer = response.get("answer")

    return answer.strip().lower() == "yes"

def process_json_data_jsonl(input_file, output_file):
    """
    Process the input JSON data and save results incrementally in JSONL format.
    """
    with open(input_file, 'r') as infile:
        data = json.load(infile)
    
    output_data = {
        "SFT": [],
        "DPO": []
    }

    # 데이터 처리
    for entry in tqdm(data):
        prompt = entry.get("prompt")
        action = entry.get("action")
        existing_reasoning = entry.get("reasoning")
        
        # Step 1: Get new reasoning from LLM
        new_reasoning = extract_reasoning(prompt, action)
        
        # Step 2: Compare existing reasoning with new reasoning using LLM
        if are_reasonings_equivalent(existing_reasoning, new_reasoning):
            # Step 3: If equivalent, add to SFT
            valid_entry = {
                "prompt": prompt,
                "existing_reasoning": existing_reasoning,
                "new_reasoning": new_reasoning,
                "action": action
            }
            output_data["SFT"].append(valid_entry)  # SFT 리스트에 추가
        else:
            # Step 4: If not equivalent, add to DPO
            valid_entry = {
                "prompt": prompt,
                "negative": existing_reasoning,
                "positive": new_reasoning,
            }
            output_data["DPO"].append(valid_entry)  # DPO 리스트에 추가




# Example usage
input_file = "/home/jihwan/NashIP/result/BR31/llama_basic_0_llama_basic.json"  # Path to your JSON file
output_file = "/home/jihwan/NashIP/result/BR31/llama_basic_0_llama_basic_refined_data.json"  # Path to save filtered data

# Process the data
process_json_data_jsonl(input_file, output_file)
