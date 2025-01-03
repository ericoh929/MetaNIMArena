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

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4, 5'

import os
import google.generativeai as genai
genai.configure(api_key=os.environ['GEMINI_API_KEY'])

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"),)



parser = argparse.ArgumentParser(description='arguments for training')

parser.add_argument('--agent1_model',     type=str,   default=None, help='model')
parser.add_argument('--agent2_model',     type=str,   default=None, help='model')
parser.add_argument('--agent1_prompt',     type=str,   default='basic', help='prompt_method')
parser.add_argument('--agent2_prompt',     type=str,   default='basic', help='prompt_method')
parser.add_argument('--look_ahead',     type=int,   default='0', help='prompt_method')
parser.add_argument('--num_games',     type=int,   default='50', help='prompt_method')

args = parser.parse_args()

# Set up your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize the game parameters
total_items = 20  # Total items in the pile (e.g., 21)
# max_take = 3  # Maximum items that can be taken per turn
num_games = args.num_games  # Number of games to play
num_refine = 3
self_consistency_count = 10  # Number of responses to use for self-consistency
n_step_lookahead = args.look_ahead  # Number of lookahead steps for n-step opponent modeling
debate_rounds = 3  # Maximum number of debate rounds

def _build_model():
    if args.agent1_model == 'llama':
        base_model = '/home/jihwan/LLM/local_model/llama3.1_8b/instruct'
    elif args.agent1_model == 'gemma':
        base_model = 'google/gemma-2-9b-it'
    elif args.agent1_model == 'qwen':
        base_model = 'Qwen/Qwen2.5-7B-Instruct'
    elif args.agent1_model == 'mistral':
        base_model = 'mistralai/Mistral-7B-Instruct-v0.3'
    
    model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map = 'auto'
        )

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    return model, tokenizer

if args.agent1_model == 'llama' or args.agent1_model == 'gemma' or args.agent1_model == 'qwen' or args.agent1_model == 'mistral':
    print(True)
    model, tokenizer = _build_model()

def convert_to_llama_format(system, instruction): #for instruct-fine-tuned model
    alpaca_format_str = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

    {system} <|eot_id|>
    <|start_header_id|>user<|end_header_id|>

    {instruction}<|eot_id|>
    <|start_header_id|>assistant<|end_header_id|>
    """
    return alpaca_format_str

# def say_model(system, instruction_str, model=model, tokenizer=tokenizer):
#     if args.agent1_model == 'llama':
#         inputs = tokenizer(convert_to_llama_format(system, instruction_str), return_tensors = "pt").to("cuda")
#         if args.agent1_prompt == 'self_consistency':
#             outputs = model.generate(**inputs, max_new_tokens = 500, use_cache = True, temperature = 0.7, top_p = 0.95, pad_token_id = tokenizer.eos_token_id)
#         else:
#             outputs = model.generate(**inputs, max_new_tokens = 500, use_cache = True, temperature = 0.7, top_p = 0.95, pad_token_id = tokenizer.eos_token_id)
#         return(tokenizer.batch_decode(outputs)[0])
    
#     elif args.agent1_model == 'gemma':
#         # <bos><start_of_turn>user
#         # Write a hello world program<end_of_turn>
#         # <start_of_turn>model
#         messages = [
#         {"role": "user", "content": f"{system}\n{instruction_str}"},
#         ]
#         prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
#         inputs = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to("cuda")
#         outputs = model.generate(input_ids=inputs, max_new_tokens=500,)
#         return(tokenizer.batch_decode(outputs)[0])

#     elif args.agent1_model == 'qwen':
#         messages = [
#         {"role": "system", "content": f"{system}"},
#         {"role": "user", "content": f"{instruction_str}"}]
#         prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
#         inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
#         generated_ids = model.generate(**inputs, max_new_tokens=500, temperature = 0.7, top_p = 0.95,)
#         generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)]
#         return(tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0])
    
#     elif args.agent1_model == 'mistral':
#         messages = [{"role": "user", "content": f"{system}\n{instruction_str}"},]
#         prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_dict=True, return_tensors="pt")
#         prompt.to("cuda")
#         outputs = model.generate(**prompt, max_new_tokens=500, temperature = 0.7, top_p = 0.95,)
#         # inputs = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to("cuda")
#         return(tokenizer.batch_decode(outputs[0], skip_special_tokens=True))

# Define agents with their respective models and prompting methods
agents = [
    {"name": "Agent 1", "model": args.agent1_model, "prompting_method": args.agent1_prompt},
    {"name": "Agent 2", "model": args.agent2_model, "prompting_method": args.agent2_prompt}
]

# Function for basic move (single-response without consistency or modeling)
def get_basic_move(agent, remaining_items, max_take, last_taken):
    if last_taken is None:
        prompt = f"""
#Game Role:\n You are {agent['name']}, a participant in a simple game.\n\n
#Objective:\n Your goal is to win the game by avoiding taking the last remaining item. The person who takes the last item loses.\n\n
#Game Rule:\n 1. There is a single pile of stones.\n
2. Players take turns to take stones.\n
3. The first player can take any number of stones, but not all the stones in the first move.\n
4. On subsequent turns, the number of stones a player can take must be at least 1 and at most twice the number of stones the previous player took.\n
5. The player who takes the last stone wins the game.\n\n
#Current State:\n There are {remaining_items} stones remaining in the pile.\n
You can take between 1 and {max_take-1} stones on your turn, where {max_take-1} = min(2 × {last_taken}, {remaining_items-1}).\n\n

#Task:\nYou are the first player. Based on the current state of the game, decide how many items you will take (between 1 and {remaining_items-1}) on this turn.\n\n

The output should be a markdown code snippet formatted in the following schema, including the leading and trailing \\`\\`\\`json" and "\\`\\`\\`":\n\n```\n{{\n\t"reasoning": string  // This is the reasons for the action\n\t"action": integer  // This is an action you take based on the reasoning. Only provide integer between 1 and {max_take-1}. You cannot choose 0.\n}}
    """
    else:
        prompt = f"""
#Game Role:\n You are {agent['name']}, a participant in a simple game.\n\n
#Objective:\n Your goal is to win the game by avoiding taking the last remaining item. The person who takes the last item loses.\n\n
#Game Rule:\n 1. There is a single pile of stones.\n
2. Players take turns to take stones.\n
3. The first player can take any number of stones, but not all the stones in the first move.\n
4. On subsequent turns, the number of stones a player can take must be at least 1 and at most twice the number of stones the previous player took.\n
5. The player who takes the last stone wins the game.\n\n
#Current State:\n There are {remaining_items} stones remaining in the pile.\n
The last player took {last_taken} stones.\n
You can take between 1 and {max_take} stones on your turn, where {max_take} = min(2 × {last_taken}, {remaining_items}).\n\n

#Task:\nBased on the current state of the game, decide how many items you will take (between 1 and {max_take}) on this turn.\n\n

The output should be a markdown code snippet formatted in the following schema, including the leading and trailing \\`\\`\\`json" and "\\`\\`\\`":\n\n```\n{{\n\t"reasoning": string  // This is the reasons for the action\n\t"action": integer  // This is an action you take based on the reasoning. Only provide integer between 1 and {max_take}. You cannot choose 0.\n}}
"""
    if agent["model"] == 'llama':
        while True:
            try:
                system_prompt = "You are a simple skil player."
                response = say_model(system_prompt, prompt)
                content = response.split("<|start_header_id|>assistant<|end_header_id|>\n")[1]
                matches_with_braces = re.search(r'\{.*?\}', content, re.DOTALL)
                parsed_content_with_braces = matches_with_braces.group(0) if matches_with_braces else None
                try: 
                    parsed_content_with_braces = parsed_content_with_braces.replace('\xa0', '').strip()       
                except (AttributeError, json.JSONDecodeError) as e:
                    print(f"Error encountered: {e}. Retrying...")
                    continue  # Retry by calling say_model again
                try:
                    parsed_content = json.loads(parsed_content_with_braces)
                except (AttributeError, json.JSONDecodeError) as e:
                    print(f"Error encountered: {e}. Retrying...")
                    continue  # Retry by calling say_model again
                break
            
            except (AttributeError, json.JSONDecodeError) as e:
                print(f"Error encountered: {e}. Retrying...")
                continue  # Retry by calling say_model again

    elif agent["model"] == 'gemma':
        while True:
            try:
                system_prompt = "You are a simple skil player."
                response = say_model(system_prompt, prompt)
                content = response.split("<start_of_turn>model\n")[1]
                matches_with_braces = re.search(r'\{.*?\}', content, re.DOTALL)
                parsed_content_with_braces = matches_with_braces.group(0) if matches_with_braces else None
                try: 
                    parsed_content_with_braces = parsed_content_with_braces.replace('\xa0', '').strip()       
                except (AttributeError, json.JSONDecodeError) as e:
                    print(f"Error encountered: {e}. Retrying...")
                    continue  # Retry by calling say_model again
                try:
                    parsed_content = json.loads(parsed_content_with_braces)
                except (AttributeError, json.JSONDecodeError) as e:
                    print(f"Error encountered: {e}. Retrying...")
                    continue  # Retry by calling say_model again
                break
            
            except (AttributeError, json.JSONDecodeError) as e:
                print(f"Error encountered: {e}. Retrying...")
                continue  # Retry by calling say_model again

    elif agent["model"] == 'gemini-1.5-flash' or agent["model"] == 'gemini-1.5-pro':

        # Create the model
        generation_config = {
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 8192,
        "response_mime_type": "application/json",
        }

        model = genai.GenerativeModel(
        model_name=agent["model"],
        generation_config=generation_config,
        system_instruction="You are a simple skil player.",
        )

        chat_session = model.start_chat(
        history=[
        ]
        )

        response = chat_session.send_message(f"{prompt}")

        content = response.text
    elif agent["model"] == 'gpt-4o' or agent["model"] == 'gpt-4o-mini' or agent["model"] == 'gpt-3.5-turbo':
        while True:
            try:
                response = client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a simple skil player.",
                    },
                    {
                        "role": "user",
                        "content": f"{prompt}",
                    }
                ],
                model=agent["model"],
                temperature=0.7
                )
                content = response.choices[0].message.content
                matches_with_braces = re.search(r'\{.*?\}', content, re.DOTALL)
                parsed_content_with_braces = matches_with_braces.group(0) if matches_with_braces else None
                parsed_content_with_braces = parsed_content_with_braces.replace('\xa0', '').strip()

                parsed_content = json.loads(parsed_content_with_braces)
                break
            except (AttributeError, json.JSONDecodeError) as e:
                print(f"Error encountered: {e}. Retrying...")
                continue  # Retry by calling say_model again

    matches_with_braces = re.search(r'\{.*?\}', content, re.DOTALL)
    parsed_content_with_braces = matches_with_braces.group(0) if matches_with_braces else None
    parsed_content_with_braces = parsed_content_with_braces.replace('\xa0', '').strip()

    parsed_content = json.loads(parsed_content_with_braces)

    reasoning = parsed_content.get("reasoning")
    action = parsed_content.get("action")

    return reasoning, action

# Function for self-consistency: generate multiple responses and choose the most common move
def get_consistent_move(agent, remaining_items, num_responses, max_take, last_taken):
    if last_taken is None:
        prompt = f"""
        #Game Role:\n You are {agent['name']}, a participant in a simple game.\n\n
        #Objective:\n Your goal is to win the game by avoiding taking the last remaining item. The person who takes the last item loses.\n\n
        #Game Rule:\n 1. There is a single pile of stones.\n
        2. Players take turns to take stones.\n
        3. The first player can take any number of stones, but not all the stones in the first move.\n
        4. On subsequent turns, the number of stones a player can take must be at least 1 and at most twice the number of stones the previous player took.\n
        5. The player who takes the last stone wins the game.\n\n
        #Current State:\n There are {remaining_items} stones remaining in the pile.\n
        You can take between 1 and {max_take-1} stones on your turn, where {max_take-1} = min(2 × {last_taken}, {remaining_items-1}).\n\n

        #Task:\nYou are the first player. Based on the current state of the game, decide how many items you will take (between 1 and {remaining_items-1}) on this turn.\n\n

        The output should be a markdown code snippet formatted in the following schema, including the leading and trailing \\`\\`\\`json" and "\\`\\`\\`":\n\n```\n{{\n\t"reasoning": string  // This is the reasons for the action\n\t"action": integer  // This is an action you take based on the reasoning. Only provide integer between 1 and {max_take-1}. You cannot choose 0.\n}}
            """
    else:
        prompt = f"""
        #Game Role:\n You are {agent['name']}, a participant in a simple game.\n\n
        #Objective:\n Your goal is to win the game by avoiding taking the last remaining item. The person who takes the last item loses.\n\n
        #Game Rule:\n 1. There is a single pile of stones.\n
        2. Players take turns to take stones.\n
        3. The first player can take any number of stones, but not all the stones in the first move.\n
        4. On subsequent turns, the number of stones a player can take must be at least 1 and at most twice the number of stones the previous player took.\n
        5. The player who takes the last stone wins the game.\n\n
        #Current State:\n There are {remaining_items} stones remaining in the pile.\n
        The last player took {last_taken} stones.\n
        You can take between 1 and {max_take} stones on your turn, where {max_take} = min(2 × {last_taken}, {remaining_items}).\n\n

        #Task:\nBased on the current state of the game, decide how many items you will take (between 1 and {max_take}) on this turn.\n\n

        The output should be a markdown code snippet formatted in the following schema, including the leading and trailing \\`\\`\\`json" and "\\`\\`\\`":\n\n```\n{{\n\t"reasoning": string  // This is the reasons for the action\n\t"action": integer  // This is an action you take based on the reasoning. Only provide integer between 1 and {max_take}. You cannot choose 0.\n}}
        """
    moves = []

    if agent["model"] == 'llama':
        for _ in range(num_responses):
            while True:
                try:
                    system_prompt = "You are a skilled Nim player."
                    response = say_model(system_prompt, prompt)
                    content = response.split("<|start_header_id|>assistant<|end_header_id|>\n")[1]
                    matches_with_braces = re.search(r'\{.*?\}', content, re.DOTALL)
                    parsed_content_with_braces = matches_with_braces.group(0) if matches_with_braces else None
                    try: 
                        parsed_content_with_braces = parsed_content_with_braces.replace('\xa0', '').strip()       
                    except (AttributeError, json.JSONDecodeError) as e:
                        print(f"Error encountered: {e}. Retrying...")
                        continue  # Retry by calling say_model again
                    try:
                        parsed_content = json.loads(parsed_content_with_braces)
                    except (AttributeError, json.JSONDecodeError) as e:
                        print(f"Error encountered: {e}. Retrying...")
                        continue  # Retry by calling say_model again
                    break
                
                except (AttributeError, json.JSONDecodeError) as e:
                    print(f"Error encountered: {e}. Retrying...")
                    continue  # Retry by calling say_model again
        
            content = response.split("<|start_header_id|>assistant<|end_header_id|>\n")[1]

            matches_with_braces = re.search(r'\{.*?\}', content, re.DOTALL)

            parsed_content_with_braces = matches_with_braces.group(0) if matches_with_braces else None
            parsed_content_with_braces = parsed_content_with_braces.replace('\xa0', '').strip()

            parsed_content = json.loads(parsed_content_with_braces)

            reasoning = parsed_content.get("reasoning")
            action = parsed_content.get("action")
            move = int(action)
            moves.append(move)

    elif agent["model"] == 'gemma':
        for _ in range(num_responses):
            while True:
                try:
                    system_prompt = "You are a skilled Nim player."
                    response = say_model(system_prompt, prompt)
                    content = response.split("<start_of_turn>model\n")[1]
                    matches_with_braces = re.search(r'\{.*?\}', content, re.DOTALL)
                    parsed_content_with_braces = matches_with_braces.group(0) if matches_with_braces else None
                    try: 
                        parsed_content_with_braces = parsed_content_with_braces.replace('\xa0', '').strip()       
                    except (AttributeError, json.JSONDecodeError) as e:
                        print(f"Error encountered: {e}. Retrying...")
                        continue  # Retry by calling say_model again
                    try:
                        parsed_content = json.loads(parsed_content_with_braces)
                    except (AttributeError, json.JSONDecodeError) as e:
                        print(f"Error encountered: {e}. Retrying...")
                        continue  # Retry by calling say_model again
                    break
                
                except (AttributeError, json.JSONDecodeError) as e:
                    print(f"Error encountered: {e}. Retrying...")
                    continue  # Retry by calling say_model again
            content = response.split("<start_of_turn>model\n")[1]

            matches_with_braces = re.search(r'\{.*?\}', content, re.DOTALL)

            parsed_content_with_braces = matches_with_braces.group(0) if matches_with_braces else None
            parsed_content_with_braces = parsed_content_with_braces.replace('\xa0', '').strip()

            parsed_content = json.loads(parsed_content_with_braces)

            reasoning = parsed_content.get("reasoning")
            action = parsed_content.get("action")
            move = int(action)
            moves.append(move)

    elif agent["model"] == 'gemini-1.5-flash' or agent["model"] == 'gemini-1.5-pro':
        for _ in range(num_responses):
            # Create the model
            generation_config = {
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 8192,
            "response_mime_type": "application/json",
            }

            model = genai.GenerativeModel(
            model_name=agent["model"],
            generation_config=generation_config,
            system_instruction="You are a skilled Nim player.",
            )

            chat_session = model.start_chat(
            history=[
            ]
            )

            response = chat_session.send_message(f"{prompt}")

            content = response.text
            matches_with_braces = re.search(r'\{.*?\}', content, re.DOTALL)

            parsed_content_with_braces = matches_with_braces.group(0) if matches_with_braces else None
            parsed_content_with_braces = parsed_content_with_braces.replace('\xa0', '').strip()

            parsed_content = json.loads(parsed_content_with_braces)

            reasoning = parsed_content.get("reasoning")
            action = parsed_content.get("action")
            move = int(action)
            moves.append(move)

    elif agent["model"] == 'gpt-4o' or agent["model"] == 'gpt-4o-mini' or agent["model"] == 'gpt-3.5-turbo':
        for _ in range(num_responses):
            while True:
                try:
                    response = client.chat.completions.create(
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a skilled Nim player.",
                        },
                        {
                            "role": "user",
                            "content": f"{prompt}",
                        }
                    ],
                    model=agent["model"],
                    temperature=0.7
                    )
                    content = response.choices[0].message.content
                    matches_with_braces = re.search(r'\{.*?\}', content, re.DOTALL)
                    parsed_content_with_braces = matches_with_braces.group(0) if matches_with_braces else None
                    parsed_content_with_braces = parsed_content_with_braces.replace('\xa0', '').strip()

                    parsed_content = json.loads(parsed_content_with_braces)
                    break
                except (AttributeError, json.JSONDecodeError) as e:
                    print(f"Error encountered: {e}. Retrying...")
                    continue  # Retry by calling say_model again

            matches_with_braces = re.search(r'\{.*?\}', content, re.DOTALL)

            parsed_content_with_braces = matches_with_braces.group(0) if matches_with_braces else None
            parsed_content_with_braces = parsed_content_with_braces.replace('\xa0', '').strip()

            parsed_content = json.loads(parsed_content_with_braces)

            reasoning = parsed_content.get("reasoning")
            action = parsed_content.get("action")
            move = int(action)
            moves.append(move)
    most_common_move = Counter(moves).most_common(1)[0][0]

    return reasoning, most_common_move

# Function for self-reflection prompting
def get_move_with_reflection(agent, remaining_items, max_take, last_taken):
    if last_taken is None:
        prompt_initial = f"""
        #Game Role:\n You are {agent['name']}, a participant in a simple game.\n\n
        #Objective:\n Your goal is to win the game by avoiding taking the last remaining item. The person who takes the last item loses.\n\n
        #Game Rule:\n 1. There is a single pile of stones.\n
        2. Players take turns to take stones.\n
        3. The first player can take any number of stones, but not all the stones in the first move.\n
        4. On subsequent turns, the number of stones a player can take must be at least 1 and at most twice the number of stones the previous player took.\n
        5. The player who takes the last stone wins the game.\n\n
        #Current State:\n There are {remaining_items} stones remaining in the pile.\n
        You can take between 1 and {max_take-1} stones on your turn, where {max_take-1} = min(2 × {last_taken}, {remaining_items-1}).\n\n

        #Task:\nYou are the first player. Based on the current state of the game, decide how many items you will take (between 1 and {remaining_items-1}) on this turn.\n\n

        The output should be a markdown code snippet formatted in the following schema, including the leading and trailing \\`\\`\\`json" and "\\`\\`\\`":\n\n```\n{{\n\t"reasoning": string  // This is the reasons for the action\n\t"action": integer  // This is an action you take based on the reasoning. Only provide integer between 1 and {max_take-1}. You cannot choose 0.\n}}
            """
    else:
        prompt_initial = f"""
        #Game Role:\n You are {agent['name']}, a participant in a simple game.\n\n
        #Objective:\n Your goal is to win the game by avoiding taking the last remaining item. The person who takes the last item loses.\n\n
        #Game Rule:\n 1. There is a single pile of stones.\n
        2. Players take turns to take stones.\n
        3. The first player can take any number of stones, but not all the stones in the first move.\n
        4. On subsequent turns, the number of stones a player can take must be at least 1 and at most twice the number of stones the previous player took.\n
        5. The player who takes the last stone wins the game.\n\n
        #Current State:\n There are {remaining_items} stones remaining in the pile.\n
        The last player took {last_taken} stones.\n
        You can take between 1 and {max_take} stones on your turn, where {max_take} = min(2 × {last_taken}, {remaining_items}).\n\n

        #Task:\nBased on the current state of the game, decide how many items you will take (between 1 and {max_take}) on this turn.\n\n

        The output should be a markdown code snippet formatted in the following schema, including the leading and trailing \\`\\`\\`json" and "\\`\\`\\`":\n\n```\n{{\n\t"reasoning": string  // This is the reasons for the action\n\t"action": integer  // This is an action you take based on the reasoning. Only provide integer between 1 and {max_take}. You cannot choose 0.\n}}
        """

    if agent["model"] == 'llama':
        while True:
            try:
                system_prompt = "You are a skilled Nim player."
                response = say_model(system_prompt, prompt_initial)
                content = response.split("<|start_header_id|>assistant<|end_header_id|>\n")[1]
                matches_with_braces = re.search(r'\{.*?\}', content, re.DOTALL)
                parsed_content_with_braces = matches_with_braces.group(0) if matches_with_braces else None
                try: 
                    parsed_content_with_braces = parsed_content_with_braces.replace('\xa0', '').strip()       
                except (AttributeError, json.JSONDecodeError) as e:
                    print(f"Error encountered: {e}. Retrying...")
                    continue  # Retry by calling say_model again
                try:
                    parsed_content = json.loads(parsed_content_with_braces)
                except (AttributeError, json.JSONDecodeError) as e:
                    print(f"Error encountered: {e}. Retrying...")
                    continue  # Retry by calling say_model again
                break
            
            except (AttributeError, json.JSONDecodeError) as e:
                print(f"Error encountered: {e}. Retrying...")
                continue  # Retry by calling say_model again

    elif agent["model"] == 'gemma':
        while True:
            try:
                system_prompt = "You are a skilled Nim player."
                response = say_model(system_prompt, prompt_initial)
                content = response.split("<start_of_turn>model\n")[1]
                matches_with_braces = re.search(r'\{.*?\}', content, re.DOTALL)
                parsed_content_with_braces = matches_with_braces.group(0) if matches_with_braces else None
                try: 
                    parsed_content_with_braces = parsed_content_with_braces.replace('\xa0', '').strip()       
                except (AttributeError, json.JSONDecodeError) as e:
                    print(f"Error encountered: {e}. Retrying...")
                    continue  # Retry by calling say_model again
                try:
                    parsed_content = json.loads(parsed_content_with_braces)
                except (AttributeError, json.JSONDecodeError) as e:
                    print(f"Error encountered: {e}. Retrying...")
                    continue  # Retry by calling say_model again
                break
            
            except (AttributeError, json.JSONDecodeError) as e:
                print(f"Error encountered: {e}. Retrying...")
                continue  # Retry by calling say_model again

    elif agent["model"] == 'gemini-1.5-flash' or agent["model"] == 'gemini-1.5-pro':

        # Create the model
        generation_config = {
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 8192,
        "response_mime_type": "application/json",
        }

        model = genai.GenerativeModel(
        model_name=agent["model"],
        generation_config=generation_config,
        system_instruction="You are a skilled Nim player.",
        )

        chat_session = model.start_chat(
        history=[
        ]
        )

        response = chat_session.send_message(f"{prompt_initial}")

        content = response.text

    elif agent["model"] == 'gpt-4o' or agent["model"] == 'gpt-4o-mini' or agent["model"] == 'gpt-3.5-turbo':
        while True:
            try:
                response = client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a skilled Nim player.",
                    },
                    {
                        "role": "user",
                        "content": f"{prompt_initial}",
                    }
                ],
                model=agent["model"],
                temperature=0.7
                )
                content = response.choices[0].message.content
                matches_with_braces = re.search(r'\{.*?\}', content, re.DOTALL)
                parsed_content_with_braces = matches_with_braces.group(0) if matches_with_braces else None
                parsed_content_with_braces = parsed_content_with_braces.replace('\xa0', '').strip()

                parsed_content = json.loads(parsed_content_with_braces)
                break
            except (AttributeError, json.JSONDecodeError) as e:
                print(f"Error encountered: {e}. Retrying...")
                continue  # Retry by calling say_model again

    matches_with_braces = re.search(r'\{.*?\}', content, re.DOTALL)

    parsed_content_with_braces = matches_with_braces.group(0) if matches_with_braces else None
    parsed_content_with_braces = parsed_content_with_braces.replace('\xa0', '').strip()
    parsed_content = json.loads(parsed_content_with_braces)

    initial_reasoning = parsed_content.get("reasoning")
    action = parsed_content.get("action")

    initial_move = int(action)

    for k in range(num_refine):
        if last_taken is None:
            feedback_prompt = f"""
            #Game Role:\n You are {agent['name']}, a participant in a simple game.\n\n
            #Objective:\n Your goal is to win the game by avoiding taking the last remaining item. The person who takes the last item loses.\n\n
            #Game Rule:\n 1. There is a single pile of stones.\n
            2. Players take turns to take stones.\n
            3. The first player can take any number of stones, but not all the stones in the first move.\n
            4. On subsequent turns, the number of stones a player can take must be at least 1 and at most twice the number of stones the previous player took.\n
            5. The player who takes the last stone wins the game.\n\n
            #Current State:\n There are {remaining_items} stones remaining in the pile.\n
            You can take between 1 and {max_take-1} stones on your turn, where {max_take-1} = min(2 × {last_taken}, {remaining_items-1}).\n\n
            #Task:\nYou are the first player. Based on the current state of the game, give a feedback on the first trial's reasoning and action.\n\n
            #First trial's reasoning and action:\nYou initially chose {initial_move} items at first trial by the reason: '{initial_reasoning}'.\n\n

            The output should be a markdown code snippet formatted in the following schema, including the leading and trailing \\`\\`\\`json" and "\\`\\`\\`":\n\n```\n{{\n\t"feedback": string  // This is the feedback for the selected action and reasoning\n}}
                """
        else:
            feedback_prompt = f"""
            #Game Role:\n You are {agent['name']}, a participant in a simple game.\n\n
            #Objective:\n Your goal is to win the game by avoiding taking the last remaining item. The person who takes the last item loses.\n\n
            #Game Rule:\n 1. There is a single pile of stones.\n
            2. Players take turns to take stones.\n
            3. The first player can take any number of stones, but not all the stones in the first move.\n
            4. On subsequent turns, the number of stones a player can take must be at least 1 and at most twice the number of stones the previous player took.\n
            5. The player who takes the last stone wins the game.\n\n
            #Current State:\n There are {remaining_items} stones remaining in the pile.\n
            The last player took {last_taken} stones.\n
            You can take between 1 and {max_take} stones on your turn, where {max_take} = min(2 × {last_taken}, {remaining_items}).\n\n
            #Task:\nBased on the current state of the game, , give a feedback on the first trial's reasoning and action.\n\n
            #First trial's reasoning and action:\nYou initially chose {initial_move} items at first trial by the reason: '{initial_reasoning}'.\n\n

            The output should be a markdown code snippet formatted in the following schema, including the leading and trailing \\`\\`\\`json" and "\\`\\`\\`":\n\n```\n{{\n\t"feedback": string  // This is the feedback for the selected action and reasoning\n}}
            """

        if agent["model"] == 'llama':
            while True:
                try:
                    system_prompt = "You are a skilled Nim player."
                    response = say_model(system_prompt, feedback_prompt)
                    content = response.split("<|start_header_id|>assistant<|end_header_id|>\n")[1]
                    matches_with_braces = re.search(r'\{.*?\}', content, re.DOTALL)
                    parsed_content_with_braces = matches_with_braces.group(0) if matches_with_braces else None
                    try: 
                        parsed_content_with_braces = parsed_content_with_braces.replace('\xa0', '').strip()       
                    except (AttributeError, json.JSONDecodeError) as e:
                        print(f"Error encountered: {e}. Retrying...")
                        continue  # Retry by calling say_model again
                    try:
                        parsed_content = json.loads(parsed_content_with_braces)
                    except (AttributeError, json.JSONDecodeError) as e:
                        print(f"Error encountered: {e}. Retrying...")
                        continue  # Retry by calling say_model again
                    break
                
                except (AttributeError, json.JSONDecodeError) as e:
                    print(f"Error encountered: {e}. Retrying...")
                    continue  # Retry by calling say_model again

        elif agent["model"] == 'gemma':
            while True:
                try:
                    system_prompt = "You are a skilled Nim player."
                    response = say_model(system_prompt, feedback_prompt)
                    content = response.split("<start_of_turn>model\n")[1]
                    matches_with_braces = re.search(r'\{.*?\}', content, re.DOTALL)
                    parsed_content_with_braces = matches_with_braces.group(0) if matches_with_braces else None
                    try: 
                        parsed_content_with_braces = parsed_content_with_braces.replace('\xa0', '').strip()       
                    except (AttributeError, json.JSONDecodeError) as e:
                        print(f"Error encountered: {e}. Retrying...")
                        continue  # Retry by calling say_model again
                    try:
                        parsed_content = json.loads(parsed_content_with_braces)
                    except (AttributeError, json.JSONDecodeError) as e:
                        print(f"Error encountered: {e}. Retrying...")
                        continue  # Retry by calling say_model again
                    break
                
                except (AttributeError, json.JSONDecodeError) as e:
                    print(f"Error encountered: {e}. Retrying...")
                    continue  # Retry by calling say_model again

        elif agent["model"] == 'gemini-1.5-flash' or agent["model"] == 'gemini-1.5-pro':

            generation_config = {
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 8192,
            "response_mime_type": "application/json",
            }

            model = genai.GenerativeModel(
            model_name=agent["model"],
            generation_config=generation_config,
            system_instruction="You are a skilled Nim player.",
            )

            chat_session = model.start_chat(
            history=[
            ]
            )

            response = chat_session.send_message(f"{feedback_prompt}")

            content = response.text

        elif agent["model"] == 'gpt-4o' or agent["model"] == 'gpt-4o-mini' or agent["model"] == 'gpt-3.5-turbo':
            while True:
                try:
                    response = client.chat.completions.create(
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a skilled Nim player.",
                        },
                        {
                            "role": "user",
                            "content": f"{feedback_prompt}",
                        }
                    ],
                    model=agent["model"],
                    temperature=0.7
                    )
                    content = response.choices[0].message.content
                    matches_with_braces = re.search(r'\{.*?\}', content, re.DOTALL)
                    parsed_content_with_braces = matches_with_braces.group(0) if matches_with_braces else None
                    parsed_content_with_braces = parsed_content_with_braces.replace('\xa0', '').strip()

                    parsed_content = json.loads(parsed_content_with_braces)
                    break
                except (AttributeError, json.JSONDecodeError) as e:
                    print(f"Error encountered: {e}. Retrying...")
                    continue  # Retry by calling say_model again
        
        matches_with_braces = re.search(r'\{.*?\}', content, re.DOTALL)

        parsed_content_with_braces = matches_with_braces.group(0) if matches_with_braces else None
        parsed_content_with_braces = parsed_content_with_braces.replace('\xa0', '').strip()
        parsed_content = json.loads(parsed_content_with_braces)

        feedback = parsed_content.get("feedback")

        if last_taken is None:
            refine_prompt = f"""
            #Game Role:\n You are {agent['name']}, a participant in a simple game.\n\n
            #Objective:\n Your goal is to win the game by avoiding taking the last remaining item. The person who takes the last item loses.\n\n
            #Game Rule:\n 1. There is a single pile of stones.\n
            2. Players take turns to take stones.\n
            3. The first player can take any number of stones, but not all the stones in the first move.\n
            4. On subsequent turns, the number of stones a player can take must be at least 1 and at most twice the number of stones the previous player took.\n
            5. The player who takes the last stone wins the game.\n\n
            #Current State:\n There are {remaining_items} stones remaining in the pile.\n
            You can take between 1 and {max_take-1} stones on your turn, where {max_take-1} = min(2 × {last_taken}, {remaining_items-1}).\n\n

            You initially chose {initial_move} items at first trial by the reason: '{initial_reasoning}'.\n\n
            You recieved feedback on your action and reasoning: {feedback}\n\n

            #Task:\nYou are the first player. Based on the current state of the game and the feedback, refine your reasoning and action. And finally, decide how many items you will take (between 1 and {remaining_items-1}) on this turn.\n\n

            The output should be a markdown code snippet formatted in the following schema, including the leading and trailing \\`\\`\\`json" and "\\`\\`\\`":\n\n```\n{{\n\t"reasoning": string  // This is the reasons for the action\n\t"action": integer  // This is an action you take based on the reasoning. Only provide integer between 1 and {max_take-1}. You cannot choose 0.\n}}
                """
        else:
            refine_prompt = f"""
            #Game Role:\n You are {agent['name']}, a participant in a simple game.\n\n
            #Objective:\n Your goal is to win the game by avoiding taking the last remaining item. The person who takes the last item loses.\n\n
            #Game Rule:\n 1. There is a single pile of stones.\n
            2. Players take turns to take stones.\n
            3. The first player can take any number of stones, but not all the stones in the first move.\n
            4. On subsequent turns, the number of stones a player can take must be at least 1 and at most twice the number of stones the previous player took.\n
            5. The player who takes the last stone wins the game.\n\n
            #Current State:\n There are {remaining_items} stones remaining in the pile.\n
            The last player took {last_taken} stones.\n
            You can take between 1 and {max_take} stones on your turn, where {max_take} = min(2 × {last_taken}, {remaining_items}).\n\n

            You initially chose {initial_move} items at first trial by the reason: '{initial_reasoning}'.\n\n
            You recieved feedback on your action and reasoning: {feedback}\n\n

            #Task:\nBased on the current state of the game and the feedback, refine your reasoning and action. And finally, decide how many items you will take (between 1 and {max_take}) on this turn.\n\n

            The output should be a markdown code snippet formatted in the following schema, including the leading and trailing \\`\\`\\`json" and "\\`\\`\\`":\n\n```\n{{\n\t"reasoning": string  // This is the reasons for the action\n\t"action": integer  // This is an action you take based on the reasoning. Only provide integer between 1 and {max_take}. You cannot choose 0.\n}}
            """

        if agent["model"] == 'llama':
            while True:
                try:
                    system_prompt = "You are a skilled Nim player."
                    response = say_model(system_prompt, refine_prompt)
                    content = response.split("<|start_header_id|>assistant<|end_header_id|>\n")[1]
                    matches_with_braces = re.search(r'\{.*?\}', content, re.DOTALL)
                    parsed_content_with_braces = matches_with_braces.group(0) if matches_with_braces else None
                    try: 
                        parsed_content_with_braces = parsed_content_with_braces.replace('\xa0', '').strip()       
                    except (AttributeError, json.JSONDecodeError) as e:
                        print(f"Error encountered: {e}. Retrying...")
                        continue  # Retry by calling say_model again
                    try:
                        parsed_content = json.loads(parsed_content_with_braces)
                    except (AttributeError, json.JSONDecodeError) as e:
                        print(f"Error encountered: {e}. Retrying...")
                        continue  # Retry by calling say_model again
                        
                    break
                
                except (AttributeError, json.JSONDecodeError) as e:
                    print(f"Error encountered: {e}. Retrying...")
                    continue  # Retry by calling say_model again

        elif agent["model"] == 'gemma':
            while True:
                try:
                    system_prompt = "You are a skilled Nim player."
                    response = say_model(system_prompt, refine_prompt)
                    content = response.split("<start_of_turn>model\n")[1]
                    matches_with_braces = re.search(r'\{.*?\}', content, re.DOTALL)
                    parsed_content_with_braces = matches_with_braces.group(0) if matches_with_braces else None
                    try: 
                        parsed_content_with_braces = parsed_content_with_braces.replace('\xa0', '').strip()       
                    except (AttributeError, json.JSONDecodeError) as e:
                        print(f"Error encountered: {e}. Retrying...")
                        continue  # Retry by calling say_model again
                    try:
                        parsed_content = json.loads(parsed_content_with_braces)
                    except (AttributeError, json.JSONDecodeError) as e:
                        print(f"Error encountered: {e}. Retrying...")
                        continue  # Retry by calling say_model again
                    break
                
                except (AttributeError, json.JSONDecodeError) as e:
                    print(f"Error encountered: {e}. Retrying...")
                    continue  # Retry by calling say_model again

        elif agent["model"] == 'gemini-1.5-flash' or agent["model"] == 'gemini-1.5-pro':

            generation_config = {
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 8192,
            "response_mime_type": "application/json",
            }

            model = genai.GenerativeModel(
            model_name=agent["model"],
            generation_config=generation_config,
            system_instruction="You are a skilled Nim player.",
            )

            chat_session = model.start_chat(
            history=[
            ]
            )

            response = chat_session.send_message(f"{refine_prompt}")

            content = response.text

        elif agent["model"] == 'gpt-4o' or agent["model"] == 'gpt-4o-mini' or agent["model"] == 'gpt-3.5-turbo':
            while True:
                try:
                    response = client.chat.completions.create(
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a skilled Nim player.",
                        },
                        {
                            "role": "user",
                            "content": f"{refine_prompt}",
                        }
                    ],
                    model=agent["model"],
                    temperature=0.7
                    )
                    content = response.choices[0].message.content
                    matches_with_braces = re.search(r'\{.*?\}', content, re.DOTALL)
                    parsed_content_with_braces = matches_with_braces.group(0) if matches_with_braces else None
                    parsed_content_with_braces = parsed_content_with_braces.replace('\xa0', '').strip()

                    parsed_content = json.loads(parsed_content_with_braces)
                    break
                except (AttributeError, json.JSONDecodeError) as e:
                    print(f"Error encountered: {e}. Retrying...")
                    continue  # Retry by calling say_model again

        matches_with_braces = re.search(r'\{.*?\}', content, re.DOTALL)

        parsed_content_with_braces = matches_with_braces.group(0) if matches_with_braces else None
        parsed_content_with_braces = parsed_content_with_braces.replace('\xa0', '').strip()
        parsed_content = json.loads(parsed_content_with_braces)

        refined_reasoning = parsed_content.get("reasoning")
        refined_action = parsed_content.get("action")

        if initial_move == int(refined_action):
            return refined_reasoning, refined_action
        else:
            initial_move = refined_action
            initial_reasoning = refined_reasoning

    return refined_reasoning, refined_action


def self_play_debate(agent1, agent2, remaining_items, n_step_lookahead, max_take, last_taken):
    initial_remaining_items = remaining_items
    initial_max_take = max_take
    initial_last_taken = last_taken
    moves = []  # Track each agent's moves for each lookahead step
    planning = ''
    for step in range(1, n_step_lookahead + 1):
        state = f"""There are {remaining_items} stones remaining in the pile."""
        if last_taken is None:
            prompt_agent1 = f"""
#Game Role:\n You are {agent1['name']}, a participant in a simple game.\n\n
#Objective:\n Your goal is to win the game by avoiding taking the last remaining item. The person who takes the last item loses.\n\n
#Game Rule:\n 1. There is a single pile of stones.\n
2. Players take turns to take stones.\n
3. The first player can take any number of stones, but not all the stones in the first move.\n
4. On subsequent turns, the number of stones a player can take must be at least 1 and at most twice the number of stones the previous player took.\n
5. The player who takes the last stone wins the game.\n\n
#Current State:\n {state}\n
You can take between 1 and {max_take-1} stones on your turn, where {max_take-1} = min(2 × {last_taken}, {remaining_items-1}).\n\n

#Task:\nYou are the first player. Based on the current state of the game, decide how many items you will take (between 1 and {remaining_items-1}) on this turn.\n\n

The output should be a markdown code snippet formatted in the following schema, including the leading and trailing \\`\\`\\`json" and "\\`\\`\\`":\n\n```\n{{\n\t"reasoning": string  // This is the reasons for the action\n\t"action": integer  // This is an action you take based on the reasoning. Only provide integer between 1 and {max_take-1}. You cannot choose 0.\n}}
    """
        else:
            prompt_agent1 = f"""
#Game Role:\n You are {agent1['name']}, a participant in a simple game.\n\n
#Objective:\n Your goal is to win the game by avoiding taking the last remaining item. The person who takes the last item loses.\n\n
#Game Rule:\n 1. There is a single pile of stones.\n
2. Players take turns to take stones.\n
3. The first player can take any number of stones, but not all the stones in the first move.\n
4. On subsequent turns, the number of stones a player can take must be at least 1 and at most twice the number of stones the previous player took.\n
5. The player who takes the last stone wins the game.\n\n
#Current State:\n {state}\n
The last player took {last_taken} stones.\n
You can take between 1 and {max_take} stones on your turn, where {max_take} = min(2 × {last_taken}, {remaining_items}).\n\n

#Task:\nBased on the current state of the game, decide how many items you will take (between 1 and {max_take}) on this turn.\n\n

The output should be a markdown code snippet formatted in the following schema, including the leading and trailing \\`\\`\\`json" and "\\`\\`\\`":\n\n```\n{{\n\t"reasoning": string  // This is the reasons for the action\n\t"action": integer  // This is an action you take based on the reasoning. Only provide integer between 1 and {max_take}. You cannot choose 0.\n}}
"""
        # print('prompt agent1:', prompt_agent1)

        if agent1["model"] == 'llama':
            while True:
                try:
                    system_prompt = "You are a skilled Nim player."
                    response = say_model(system_prompt, prompt_agent1)
                    content = response.split("<|start_header_id|>assistant<|end_header_id|>\n")[1]
                    matches_with_braces = re.search(r'\{.*?\}', content, re.DOTALL)
                    parsed_content_with_braces = matches_with_braces.group(0) if matches_with_braces else None
                    try: 
                        parsed_content_with_braces = parsed_content_with_braces.replace('\xa0', '').strip()       
                    except (AttributeError, json.JSONDecodeError) as e:
                        print(f"Error encountered: {e}. Retrying...")
                        continue  # Retry by calling say_model again
                    try:
                        parsed_content = json.loads(parsed_content_with_braces)
                    except (AttributeError, json.JSONDecodeError) as e:
                        print(f"Error encountered: {e}. Retrying...")
                        continue  # Retry by calling say_model again
                    break
                
                except (AttributeError, json.JSONDecodeError) as e:
                    print(f"Error encountered: {e}. Retrying...")
                    continue  # Retry by calling say_model again

        elif agent1["model"] == 'gemma':
            while True:
                try:
                    system_prompt = "You are a skilled Nim player."
                    response = say_model(system_prompt, prompt_agent1)
                    content = response.split("<start_of_turn>model\n")[1]
                    matches_with_braces = re.search(r'\{.*?\}', content, re.DOTALL)
                    parsed_content_with_braces = matches_with_braces.group(0) if matches_with_braces else None
                    try: 
                        parsed_content_with_braces = parsed_content_with_braces.replace('\xa0', '').strip()       
                    except (AttributeError, json.JSONDecodeError) as e:
                        print(f"Error encountered: {e}. Retrying...")
                        continue  # Retry by calling say_model again
                    try:
                        parsed_content = json.loads(parsed_content_with_braces)
                    except (AttributeError, json.JSONDecodeError) as e:
                        print(f"Error encountered: {e}. Retrying...")
                        continue  # Retry by calling say_model again
                    break
                
                except (AttributeError, json.JSONDecodeError) as e:
                    print(f"Error encountered: {e}. Retrying...")
                    continue  # Retry by calling say_model again

        elif agent1["model"] == 'gemini-1.5-flash' or agent1["model"] == 'gemini-1.5-pro':
            
            generation_config = {
            "temperature": 1,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 8192,
            "response_mime_type": "application/json",
            }

            model = genai.GenerativeModel(
            model_name=agent1["model"],
            generation_config=generation_config,
            system_instruction="You are a skilled Nim player.",
            )

            chat_session = model.start_chat(
            history=[
            ]
            )

            response = chat_session.send_message(f"{prompt_agent1}")

            content = response.text

        elif agent1["model"] == 'gpt-4o' or agent1["model"] == 'gpt-4o-mini' or agent1["model"] == 'gpt-3.5-turbo':
            while True:
                try:
                    response = client.chat.completions.create(
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a skilled Nim player.",
                        },
                        {
                            "role": "user",
                            "content": f"{prompt_agent1}",
                        }
                    ],
                    model=agent1["model"],
                    temperature=1
                    )
                    content = response.choices[0].message.content
                    matches_with_braces = re.search(r'\{.*?\}', content, re.DOTALL)
                    parsed_content_with_braces = matches_with_braces.group(0) if matches_with_braces else None
                    parsed_content_with_braces = parsed_content_with_braces.replace('\xa0', '').strip()

                    parsed_content = json.loads(parsed_content_with_braces)
                    break
                except (AttributeError, json.JSONDecodeError) as e:
                    print(f"Error encountered: {e}. Retrying...")
                    continue  # Retry by calling say_model again

        matches_with_braces = re.search(r'\{.*?\}', content, re.DOTALL)
        

        parsed_content_with_braces = matches_with_braces.group(0) if matches_with_braces else None
        parsed_content_with_braces = parsed_content_with_braces.replace('\xa0', '').strip()
        
        parsed_content = json.loads(parsed_content_with_braces)
        agent1_reasoning = parsed_content.get("reasoning")
        agent1_action = parsed_content.get("action")

        agent1_move = int(agent1_action)

        planning += f'State: {state}\n'
        planning += f'My reasoning: {agent1_reasoning}\n'
        planning += f'My action: {agent1_action}\n'

        # print('My reasoning:', agent1_reasoning)
        # print('My action:', agent1_action)
        
        remaining_items -= agent1_move
        planning += f'Remaining total items by your action: {remaining_items}\n\n'
        moves.append((agent1["name"], agent1_move, remaining_items))
        
        # Check if game ends with Agent 1's move
        if remaining_items <= 0 and step == 1:
            # print('################################################################################################################')
            return agent1_reasoning, agent1_move  # Agent 1 wins if no items remain
        if remaining_items <= 0:
            break
        last_taken = agent1_move
        max_take = min(2 * last_taken, remaining_items)
        # Agent 2's simulated response
        state = f"""There are {remaining_items} stones remaining in the pile."""
        if last_taken is None:
            prompt_agent2 = f"""
#Game Role:\n You are {agent2['name']}, a participant in a simple game.\n\n
#Objective:\n Your goal is to win the game by avoiding taking the last remaining item. The person who takes the last item loses.\n\n
#Game Rule:\n 1. There is a single pile of stones.\n
2. Players take turns to take stones.\n
3. The first player can take any number of stones, but not all the stones in the first move.\n
4. On subsequent turns, the number of stones a player can take must be at least 1 and at most twice the number of stones the previous player took.\n
5. The player who takes the last stone wins the game.\n\n
#Current State:\n {state}\n
You can take between 1 and {max_take-1} stones on your turn, where {max_take-1} = min(2 × {last_taken}, {remaining_items-1}).\n\n

#Task:\nYou are the first player. Based on the current state of the game, decide how many items you will take (between 1 and {remaining_items-1}) on this turn.\n\n

The output should be a markdown code snippet formatted in the following schema, including the leading and trailing \\`\\`\\`json" and "\\`\\`\\`":\n\n```\n{{\n\t"reasoning": string  // This is the reasons for the action\n\t"action": integer  // This is an action you take based on the reasoning. Only provide integer between 1 and {max_take-1}. You cannot choose 0.\n}}
    """
        else:
            prompt_agent2 = f"""
#Game Role:\n You are {agent2['name']}, a participant in a simple game.\n\n
#Objective:\n Your goal is to win the game by avoiding taking the last remaining item. The person who takes the last item loses.\n\n
#Game Rule:\n 1. There is a single pile of stones.\n
2. Players take turns to take stones.\n
3. The first player can take any number of stones, but not all the stones in the first move.\n
4. On subsequent turns, the number of stones a player can take must be at least 1 and at most twice the number of stones the previous player took.\n
5. The player who takes the last stone wins the game.\n\n
#Current State:\n {state}\n
The last player took {last_taken} stones.\n
You can take between 1 and {max_take} stones on your turn, where {max_take} = min(2 × {last_taken}, {remaining_items}).\n\n

#Task:\nBased on the current state of the game, decide how many items you will take (between 1 and {max_take}) on this turn.\n\n

The output should be a markdown code snippet formatted in the following schema, including the leading and trailing \\`\\`\\`json" and "\\`\\`\\`":\n\n```\n{{\n\t"reasoning": string  // This is the reasons for the action\n\t"action": integer  // This is an action you take based on the reasoning. Only provide integer between 1 and {max_take}. You cannot choose 0.\n}}
"""
        
        # print('prompt agent2:', prompt_agent2)
        if agent1["model"] == 'llama':
            while True:
                try:
                    system_prompt = "You are a skilled Nim player."
                    response = say_model(system_prompt, prompt_agent2)
                    content = response.split("<|start_header_id|>assistant<|end_header_id|>\n")[1]
                    matches_with_braces = re.search(r'\{.*?\}', content, re.DOTALL)
                    parsed_content_with_braces = matches_with_braces.group(0) if matches_with_braces else None
                    try: 
                        parsed_content_with_braces = parsed_content_with_braces.replace('\xa0', '').strip()       
                    except (AttributeError, json.JSONDecodeError) as e:
                        print(f"Error encountered: {e}. Retrying...")
                        continue  # Retry by calling say_model again
                    try:
                        parsed_content = json.loads(parsed_content_with_braces)
                    except (AttributeError, json.JSONDecodeError) as e:
                        print(f"Error encountered: {e}. Retrying...")
                        continue  # Retry by calling say_model again
                    break
                
                except (AttributeError, json.JSONDecodeError) as e:
                    print(f"Error encountered: {e}. Retrying...")
                    continue  # Retry by calling say_model again

        elif agent1["model"] == 'gemma':
            while True:
                try:
                    system_prompt = "You are a skilled Nim player."
                    response = say_model(system_prompt, prompt_agent2)
                    content = response.split("<start_of_turn>model\n")[1]
                    matches_with_braces = re.search(r'\{.*?\}', content, re.DOTALL)
                    parsed_content_with_braces = matches_with_braces.group(0) if matches_with_braces else None
                    try: 
                        parsed_content_with_braces = parsed_content_with_braces.replace('\xa0', '').strip()       
                    except (AttributeError, json.JSONDecodeError) as e:
                        print(f"Error encountered: {e}. Retrying...")
                        continue  # Retry by calling say_model again
                    try:
                        parsed_content = json.loads(parsed_content_with_braces)
                    except (AttributeError, json.JSONDecodeError) as e:
                        print(f"Error encountered: {e}. Retrying...")
                        continue  # Retry by calling say_model again
                    break
                
                except (AttributeError, json.JSONDecodeError) as e:
                    print(f"Error encountered: {e}. Retrying...")
                    continue  # Retry by calling say_model again

        elif agent1["model"] == 'gemini-1.5-flash' or agent1["model"] == 'gemini-1.5-pro':

            # Create the model
            generation_config = {
            "temperature": 1,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 8192,
            "response_mime_type": "application/json",
            }

            model = genai.GenerativeModel(
            model_name=agent1["model"],
            generation_config=generation_config,
            system_instruction="You are a skilled Nim player.",
            )

            chat_session = model.start_chat(
            history=[
            ]
            )

            response = chat_session.send_message(f"{prompt_agent2}")

            content = response.text

        elif agent1["model"] == 'gpt-4o' or agent1["model"] == 'gpt-4o-mini' or agent1["model"] == 'gpt-3.5-turbo':
            while True:
                try:
                    response = client.chat.completions.create(
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a skilled Nim player.",
                        },
                        {
                            "role": "user",
                            "content": f"{prompt_agent2}",
                        }
                    ],
                    model=agent1["model"],
                    temperature=1
                    )
                    content = response.choices[0].message.content
                    matches_with_braces = re.search(r'\{.*?\}', content, re.DOTALL)
                    parsed_content_with_braces = matches_with_braces.group(0) if matches_with_braces else None
                    parsed_content_with_braces = parsed_content_with_braces.replace('\xa0', '').strip()

                    parsed_content = json.loads(parsed_content_with_braces)
                    break
                except (AttributeError, json.JSONDecodeError) as e:
                    print(f"Error encountered: {e}. Retrying...")
                    continue  # Retry by calling say_model again

        matches_with_braces = re.search(r'\{.*?\}', content, re.DOTALL)

        parsed_content_with_braces = matches_with_braces.group(0) if matches_with_braces else None
        parsed_content_with_braces = parsed_content_with_braces.replace('\xa0', '').strip()
        parsed_content = json.loads(parsed_content_with_braces)

        # agent2_reasoning, agent2_action = get_move_with_debate(agent1, agent1, remaining_items, max_take, last_taken)

        agent2_reasoning = parsed_content.get("reasoning")
        agent2_action = parsed_content.get("action")

        agent2_move = int(agent2_action)

        planning += f'State: {state}\n'
        planning += f'Opponent reasoning: {agent2_reasoning}\n'
        planning += f'Opponent action: {agent2_action}\n'
        remaining_items -= agent2_move

        # print('opponent reasoning:', agent2_reasoning)
        # print('opponent action:', agent2_action)

        planning += f'Remaining total items by opponent\'s action: {remaining_items}\n\n'
        moves.append((agent2["name"], agent2_move, remaining_items))
        
        last_taken = agent2_move
        max_take = min(2 * last_taken, remaining_items)
        # Check if game ends with Agent 2's move
        if remaining_items <= 0:
            break
            # return agent1_reasoning, agent1_move  # Agent 1's initial move if Agent 2 would win

    # Final decision for Agent 1 based on the full n-step lookahead sequence

    move_sequence_str = "; ".join([f"{name} took {move} items and {remains} items remained" for name, move, remains in moves]) #Decide how many items to take between 1 and {max_take} at this current step to win by taking all remaining items on your turn, leaving no items for your opponent. Provide your reasoning and action in the following schema:
    state = f"""There are {initial_remaining_items} stones remaining in the pile."""
    if initial_last_taken is None:
        final_prompt_agent1 = f"""
#Game Role:\n You are {agent1['name']}, a participant in a simple game.\n\n
#Objective:\n Your goal is to win the game by avoiding taking the last remaining item. The person who takes the last item loses.\n\n
#Game Rule:\n 1. There is a single pile of stones.\n
2. Players take turns to take stones.\n
3. The first player can take any number of stones, but not all the stones in the first move.\n
4. On subsequent turns, the number of stones a player can take must be at least 1 and at most twice the number of stones the previous player took.\n
5. The player who takes the last stone wins the game.\n\n
Current State:\n {state}\n
You can take between 1 and {initial_max_take-1} stones on your turn, where {initial_max_take-1} = min(2 × {initial_last_taken}, {initial_remaining_items-1}).\n\n

#Task:\nYou are the first player. Based on the current state of the game, decide how many items you will take (between 1 and {initial_remaining_items-1}) on this turn.\n\n

As part of your strategy, you conducted a simulated planning process. This planning predicted possible moves by the opponent and future scenarios based on the current state of the game.
The planning results are provided below as a reference:\n
#Simulated Planning History:\n{planning}\nSimultion ends.\n\n

Now, carefully review the simulated planning history and reflect and decide how many items you will take (between 1 and {initial_remaining_items-1}) on this turn.\n

The output should be a markdown code snippet formatted in the following schema, including the leading and trailing \\`\\`\\`json" and "\\`\\`\\`":\n\n```\n{{\n\t"reasoning": string  // This is the reasons for the action\n\t"action": integer  // This is an action you take based on the reasoning. Only provide integer between 1 and {initial_max_take-1}. You cannot choose 0.\n}}
    """
    else:
        final_prompt_agent1 = f"""
#Game Role:\n You are {agent1['name']}, a participant in a simple game.\n\n
#Objective:\n Your goal is to win the game by avoiding taking the last remaining item. The person who takes the last item loses.\n\n
#Game Rule:\n 1. There is a single pile of stones.\n
2. Players take turns to take stones.\n
3. The first player can take any number of stones, but not all the stones in the first move.\n
4. On subsequent turns, the number of stones a player can take must be at least 1 and at most twice the number of stones the previous player took.\n
5. The player who takes the last stone wins the game.\n\n
#Current State:\n {state}\n
The last player took {initial_last_taken} stones.\n
You can take between 1 and {initial_max_take} stones on your turn, where {initial_max_take} = min(2 × {initial_last_taken}, {initial_remaining_items}).\n\n

#Task:\nBased on the current state of the game, decide how many items you will take (between 1 and {initial_max_take}) on this turn.\n\n

As part of your strategy, you conducted a simulated planning process. This planning predicted possible moves by the opponent and future scenarios based on the current state of the game.
The planning results are provided below as a reference:\n
#Simulated Planning History:\n{planning}\nSimultion ends.\n\n

Now, carefully review the simulated planning history and reflect and decide how many items you will take (between 1 and {initial_max_take}) on this turn.\n

The output should be a markdown code snippet formatted in the following schema, including the leading and trailing \\`\\`\\`json" and "\\`\\`\\`":\n\n```\n{{\n\t"reasoning": string  // This is the reasons for the action\n\t"action": integer  // This is an action you take based on the reasoning. Only provide integer between 1 and {initial_max_take}. You cannot choose 0.\n}}
        """
    # print('final prompt', final_prompt_agent1)
    if agent1["model"] == 'llama':
        while True:
            try:
                system_prompt = "You are a skilled Nim player."
                response = say_model(system_prompt, final_prompt_agent1)
                content = response.split("<|start_header_id|>assistant<|end_header_id|>\n")[1]
                matches_with_braces = re.search(r'\{.*?\}', content, re.DOTALL)
                parsed_content_with_braces = matches_with_braces.group(0) if matches_with_braces else None
                try: 
                    parsed_content_with_braces = parsed_content_with_braces.replace('\xa0', '').strip()       
                except (AttributeError, json.JSONDecodeError) as e:
                    print(f"Error encountered: {e}. Retrying...")
                    continue  # Retry by calling say_model again
                try:
                    parsed_content = json.loads(parsed_content_with_braces)
                except (AttributeError, json.JSONDecodeError) as e:
                    print(f"Error encountered: {e}. Retrying...")
                    continue  # Retry by calling say_model again
                break
            
            except (AttributeError, json.JSONDecodeError) as e:
                print(f"Error encountered: {e}. Retrying...")
                continue  # Retry by calling say_model again

    elif agent1["model"] == 'gemma':
        while True:
            try:
                system_prompt = "You are a skilled Nim player."
                response = say_model(system_prompt, final_prompt_agent1)
                content = response.split("<start_of_turn>model\n")[1]
                matches_with_braces = re.search(r'\{.*?\}', content, re.DOTALL)
                parsed_content_with_braces = matches_with_braces.group(0) if matches_with_braces else None
                try: 
                    parsed_content_with_braces = parsed_content_with_braces.replace('\xa0', '').strip()       
                except (AttributeError, json.JSONDecodeError) as e:
                    print(f"Error encountered: {e}. Retrying...")
                    continue  # Retry by calling say_model again
                try:
                    parsed_content = json.loads(parsed_content_with_braces)
                except (AttributeError, json.JSONDecodeError) as e:
                    print(f"Error encountered: {e}. Retrying...")
                    continue  # Retry by calling say_model again
                break
            
            except (AttributeError, json.JSONDecodeError) as e:
                print(f"Error encountered: {e}. Retrying...")
                continue  # Retry by calling say_model again

    elif agent1["model"] == 'gemini-1.5-flash' or agent1["model"] == 'gemini-1.5-pro':

        # Create the model
        generation_config = {
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 8192,
        "response_mime_type": "application/json",
        }

        model = genai.GenerativeModel(
        model_name=agent1["model"],
        generation_config=generation_config,
        system_instruction="You are a skilled Nim player.",
        )

        chat_session = model.start_chat(
        history=[
        ]
        )

        response = chat_session.send_message(f"{final_prompt_agent1}")

        content = response.text
    
    elif agent1["model"] == 'gpt-4o' or agent1["model"] == 'gpt-4o-mini' or agent1["model"] == 'gpt-3.5-turbo':
        while True:
            try:
                response = client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a skilled Nim player.",
                    },
                    {
                        "role": "user",
                        "content": f"{final_prompt_agent1}",
                    }
                ],
                model=agent1["model"],
                temperature=1
                )
                content = response.choices[0].message.content
                matches_with_braces = re.search(r'\{.*?\}', content, re.DOTALL)
                parsed_content_with_braces = matches_with_braces.group(0) if matches_with_braces else None
                parsed_content_with_braces = parsed_content_with_braces.replace('\xa0', '').strip()

                parsed_content = json.loads(parsed_content_with_braces)
                break
            except (AttributeError, json.JSONDecodeError) as e:
                print(f"Error encountered: {e}. Retrying...")
                continue  # Retry by calling say_model agai

    matches_with_braces = re.search(r'\{.*?\}', content, re.DOTALL)

    parsed_content_with_braces = matches_with_braces.group(0) if matches_with_braces else None
    parsed_content_with_braces = parsed_content_with_braces.replace('\xa0', '').strip()
    parsed_content = json.loads(parsed_content_with_braces)

    agent1_reasoning = parsed_content.get("reasoning")
    agent1_action = parsed_content.get("action")
    # print('My reasoning:', agent1_reasoning)
    # print('My action:', agent1_action)
    # print('################################################################################################################')

    agent1_final_move = int(agent1_action)
    
    return agent1_reasoning, agent1_final_move

def self_play_debate_exp(agent1, agent2, remaining_items, n_step_lookahead, max_take, last_taken):
    initial_remaining_items = remaining_items
    initial_max_take = max_take
    initial_last_taken = last_taken
    moves = []  # Track each agent's moves for each lookahead step
    planning = ''
    for step in range(1, n_step_lookahead + 1):
        state = f"""There are {remaining_items} stones remaining in the pile."""
        if last_taken is None:
            prompt_agent1 = f"""
#Game Role:\n You are {agent1['name']}, a participant in a simple game.\n\n
#Objective:\n Your goal is to win the game by avoiding taking the last remaining item. The person who takes the last item loses.\n\n
#Game Rule:\n 1. There is a single pile of stones.\n
2. Players take turns to take stones.\n
3. The first player can take any number of stones, but not all the stones in the first move.\n
4. On subsequent turns, the number of stones a player can take must be at least 1 and at most twice the number of stones the previous player took.\n
5. The player who takes the last stone wins the game.\n\n
#Current State:\n {state}\n
You can take between 1 and {max_take-1} stones on your turn, where {max_take-1} = min(2 × {last_taken}, {remaining_items-1}).\n\n

#Task:\nYou are the first player. Based on the current state of the game, decide how many items you will take (between 1 and {remaining_items-1}) on this turn.\n\n

The output should be a markdown code snippet formatted in the following schema, including the leading and trailing \\`\\`\\`json" and "\\`\\`\\`":\n\n```\n{{\n\t"reasoning": string  // This is the reasons for the action\n\t"action": integer  // This is an action you take based on the reasoning. Only provide integer between 1 and {max_take-1}. You cannot choose 0.\n}}
    """
        else:
            prompt_agent1 = f"""
#Game Role:\n You are {agent1['name']}, a participant in a simple game.\n\n
#Objective:\n Your goal is to win the game by avoiding taking the last remaining item. The person who takes the last item loses.\n\n
#Game Rule:\n 1. There is a single pile of stones.\n
2. Players take turns to take stones.\n
3. The first player can take any number of stones, but not all the stones in the first move.\n
4. On subsequent turns, the number of stones a player can take must be at least 1 and at most twice the number of stones the previous player took.\n
5. The player who takes the last stone wins the game.\n\n
#Current State:\n {state}\n
The last player took {last_taken} stones.\n
You can take between 1 and {max_take} stones on your turn, where {max_take} = min(2 × {last_taken}, {remaining_items}).\n\n

#Task:\nBased on the current state of the game, decide how many items you will take (between 1 and {max_take}) on this turn.\n\n

The output should be a markdown code snippet formatted in the following schema, including the leading and trailing \\`\\`\\`json" and "\\`\\`\\`":\n\n```\n{{\n\t"reasoning": string  // This is the reasons for the action\n\t"action": integer  // This is an action you take based on the reasoning. Only provide integer between 1 and {max_take}. You cannot choose 0.\n}}
"""
        # print('prompt agent1:', prompt_agent1)

        if agent1["model"] == 'llama':
            while True:
                try:
                    system_prompt = "You are a skilled Nim player."
                    response = say_model(system_prompt, prompt_agent1)
                    content = response.split("<|start_header_id|>assistant<|end_header_id|>\n")[1]
                    matches_with_braces = re.search(r'\{.*?\}', content, re.DOTALL)
                    parsed_content_with_braces = matches_with_braces.group(0) if matches_with_braces else None
                    try: 
                        parsed_content_with_braces = parsed_content_with_braces.replace('\xa0', '').strip()       
                    except (AttributeError, json.JSONDecodeError) as e:
                        print(f"Error encountered: {e}. Retrying...")
                        continue  # Retry by calling say_model again
                    try:
                        parsed_content = json.loads(parsed_content_with_braces)
                    except (AttributeError, json.JSONDecodeError) as e:
                        print(f"Error encountered: {e}. Retrying...")
                        continue  # Retry by calling say_model again
                    break
                
                except (AttributeError, json.JSONDecodeError) as e:
                    print(f"Error encountered: {e}. Retrying...")
                    continue  # Retry by calling say_model again

        elif agent1["model"] == 'gemma':
            while True:
                try:
                    system_prompt = "You are a skilled Nim player."
                    response = say_model(system_prompt, prompt_agent1)
                    content = response.split("<start_of_turn>model\n")[1]
                    matches_with_braces = re.search(r'\{.*?\}', content, re.DOTALL)
                    parsed_content_with_braces = matches_with_braces.group(0) if matches_with_braces else None
                    try: 
                        parsed_content_with_braces = parsed_content_with_braces.replace('\xa0', '').strip()       
                    except (AttributeError, json.JSONDecodeError) as e:
                        print(f"Error encountered: {e}. Retrying...")
                        continue  # Retry by calling say_model again
                    try:
                        parsed_content = json.loads(parsed_content_with_braces)
                    except (AttributeError, json.JSONDecodeError) as e:
                        print(f"Error encountered: {e}. Retrying...")
                        continue  # Retry by calling say_model again
                    break
                
                except (AttributeError, json.JSONDecodeError) as e:
                    print(f"Error encountered: {e}. Retrying...")
                    continue  # Retry by calling say_model again

        elif agent1["model"] == 'gemini-1.5-flash' or agent1["model"] == 'gemini-1.5-pro':
            
            generation_config = {
            "temperature": 1,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 8192,
            "response_mime_type": "application/json",
            }

            model = genai.GenerativeModel(
            model_name=agent1["model"],
            generation_config=generation_config,
            system_instruction="You are a skilled Nim player.",
            )

            chat_session = model.start_chat(
            history=[
            ]
            )

            response = chat_session.send_message(f"{prompt_agent1}")

            content = response.text

        elif agent1["model"] == 'gpt-4o' or agent1["model"] == 'gpt-4o-mini' or agent1["model"] == 'gpt-3.5-turbo':
            while True:
                try:
                    response = client.chat.completions.create(
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a skilled Nim player.",
                        },
                        {
                            "role": "user",
                            "content": f"{prompt_agent1}",
                        }
                    ],
                    model=agent1["model"],
                    temperature=1
                    )
                    content = response.choices[0].message.content
                    matches_with_braces = re.search(r'\{.*?\}', content, re.DOTALL)
                    parsed_content_with_braces = matches_with_braces.group(0) if matches_with_braces else None
                    parsed_content_with_braces = parsed_content_with_braces.replace('\xa0', '').strip()

                    parsed_content = json.loads(parsed_content_with_braces)
                    break
                except (AttributeError, json.JSONDecodeError) as e:
                    print(f"Error encountered: {e}. Retrying...")
                    continue  # Retry by calling say_model again

        matches_with_braces = re.search(r'\{.*?\}', content, re.DOTALL)
        

        parsed_content_with_braces = matches_with_braces.group(0) if matches_with_braces else None
        parsed_content_with_braces = parsed_content_with_braces.replace('\xa0', '').strip()
        
        parsed_content = json.loads(parsed_content_with_braces)
        agent1_reasoning = parsed_content.get("reasoning")
        agent1_action = parsed_content.get("action")

        agent1_move = int(agent1_action)

        planning += f'State: {state}\n'
        planning += f'My reasoning: {agent1_reasoning}\n'
        planning += f'My action: {agent1_action}\n'

        # print('My reasoning:', agent1_reasoning)
        # print('My action:', agent1_action)
        
        remaining_items -= agent1_move
        planning += f'Remaining total items by your action: {remaining_items}\n\n'
        moves.append((agent1["name"], agent1_move, remaining_items))
        
        # Check if game ends with Agent 1's move
        if remaining_items <= 0 and step == 1:
            # print('################################################################################################################')
            return agent1_reasoning, agent1_move  # Agent 1 wins if no items remain
        if remaining_items <= 0:
            break
        last_taken = agent1_move
        max_take = min(2 * last_taken, remaining_items)
        # Agent 2's simulated response
        state = f"""There are {remaining_items} stones remaining in the pile."""
        if last_taken is None:
            prompt_agent2 = f"""
#Game Role:\n You are {agent2['name']}, a participant in a simple game.\n\n
#Objective:\n Your goal is to win the game by avoiding taking the last remaining item. The person who takes the last item loses.\n\n
#Game Rule:\n 1. There is a single pile of stones.\n
2. Players take turns to take stones.\n
3. The first player can take any number of stones, but not all the stones in the first move.\n
4. On subsequent turns, the number of stones a player can take must be at least 1 and at most twice the number of stones the previous player took.\n
5. The player who takes the last stone wins the game.\n\n
#Current State:\n {state}\n
You can take between 1 and {max_take-1} stones on your turn, where {max_take-1} = min(2 × {last_taken}, {remaining_items-1}).\n\n

#Task:\nYou are the first player. Based on the current state of the game, decide how many items you will take (between 1 and {remaining_items-1}) on this turn.\n\n

The output should be a markdown code snippet formatted in the following schema, including the leading and trailing \\`\\`\\`json" and "\\`\\`\\`":\n\n```\n{{\n\t"reasoning": string  // This is the reasons for the action\n\t"action": integer  // This is an action you take based on the reasoning. Only provide integer between 1 and {max_take-1}. You cannot choose 0.\n}}
    """
        else:
            prompt_agent2 = f"""
#Game Role:\n You are {agent2['name']}, a participant in a simple game.\n\n
#Objective:\n Your goal is to win the game by avoiding taking the last remaining item. The person who takes the last item loses.\n\n
#Game Rule:\n 1. There is a single pile of stones.\n
2. Players take turns to take stones.\n
3. The first player can take any number of stones, but not all the stones in the first move.\n
4. On subsequent turns, the number of stones a player can take must be at least 1 and at most twice the number of stones the previous player took.\n
5. The player who takes the last stone wins the game.\n\n
#Current State:\n {state}\n
The last player took {last_taken} stones.\n
You can take between 1 and {max_take} stones on your turn, where {max_take} = min(2 × {last_taken}, {remaining_items}).\n\n

#Task:\nBased on the current state of the game, decide how many items you will take (between 1 and {max_take}) on this turn.\n\n

The output should be a markdown code snippet formatted in the following schema, including the leading and trailing \\`\\`\\`json" and "\\`\\`\\`":\n\n```\n{{\n\t"reasoning": string  // This is the reasons for the action\n\t"action": integer  // This is an action you take based on the reasoning. Only provide integer between 1 and {max_take}. You cannot choose 0.\n}}
"""
        
        # print('prompt agent2:', prompt_agent2)
        

        agent2_reasoning, agent2_action = get_move_with_debate(agent1, agent1, remaining_items, max_take, last_taken)


        agent2_move = int(agent2_action)

        planning += f'State: {state}\n'
        planning += f'Opponent reasoning: {agent2_reasoning}\n'
        planning += f'Opponent action: {agent2_action}\n'
        remaining_items -= agent2_move

        # print('opponent reasoning:', agent2_reasoning)
        # print('opponent action:', agent2_action)

        planning += f'Remaining total items by opponent\'s action: {remaining_items}\n\n'
        moves.append((agent2["name"], agent2_move, remaining_items))
        
        last_taken = agent2_move
        max_take = min(2 * last_taken, remaining_items)
        # Check if game ends with Agent 2's move
        if remaining_items <= 0:
            break
            # return agent1_reasoning, agent1_move  # Agent 1's initial move if Agent 2 would win

    # Final decision for Agent 1 based on the full n-step lookahead sequence

    move_sequence_str = "; ".join([f"{name} took {move} items and {remains} items remained" for name, move, remains in moves]) #Decide how many items to take between 1 and {max_take} at this current step to win by taking all remaining items on your turn, leaving no items for your opponent. Provide your reasoning and action in the following schema:
    state = f"""There are {initial_remaining_items} stones remaining in the pile."""
    if initial_last_taken is None:
        final_prompt_agent1 = f"""
#Game Role:\n You are {agent1['name']}, a participant in a simple game.\n\n
#Objective:\n Your goal is to win the game by avoiding taking the last remaining item. The person who takes the last item loses.\n\n
#Game Rule:\n 1. There is a single pile of stones.\n
2. Players take turns to take stones.\n
3. The first player can take any number of stones, but not all the stones in the first move.\n
4. On subsequent turns, the number of stones a player can take must be at least 1 and at most twice the number of stones the previous player took.\n
5. The player who takes the last stone wins the game.\n\n
Current State:\n {state}\n
You can take between 1 and {initial_max_take-1} stones on your turn, where {initial_max_take-1} = min(2 × {initial_last_taken}, {initial_remaining_items-1}).\n\n

#Task:\nYou are the first player. Based on the current state of the game, decide how many items you will take (between 1 and {initial_remaining_items-1}) on this turn.\n\n

As part of your strategy, you conducted a simulated planning process. This planning predicted possible moves by the opponent and future scenarios based on the current state of the game.
The planning results are provided below as a reference:\n
#Simulated Planning History:\n{planning}\nSimultion ends.\n\n

Now, carefully review the simulated planning history and reflect and decide how many items you will take (between 1 and {initial_remaining_items-1}) on this turn.\n

The output should be a markdown code snippet formatted in the following schema, including the leading and trailing \\`\\`\\`json" and "\\`\\`\\`":\n\n```\n{{\n\t"reasoning": string  // This is the reasons for the action\n\t"action": integer  // This is an action you take based on the reasoning. Only provide integer between 1 and {initial_max_take-1}. You cannot choose 0.\n}}
    """
    else:
        final_prompt_agent1 = f"""
#Game Role:\n You are {agent1['name']}, a participant in a simple game.\n\n
#Objective:\n Your goal is to win the game by avoiding taking the last remaining item. The person who takes the last item loses.\n\n
#Game Rule:\n 1. There is a single pile of stones.\n
2. Players take turns to take stones.\n
3. The first player can take any number of stones, but not all the stones in the first move.\n
4. On subsequent turns, the number of stones a player can take must be at least 1 and at most twice the number of stones the previous player took.\n
5. The player who takes the last stone wins the game.\n\n
#Current State:\n {state}\n
The last player took {initial_last_taken} stones.\n
You can take between 1 and {initial_max_take} stones on your turn, where {initial_max_take} = min(2 × {initial_last_taken}, {initial_remaining_items}).\n\n

#Task:\nBased on the current state of the game, decide how many items you will take (between 1 and {initial_max_take}) on this turn.\n\n

As part of your strategy, you conducted a simulated planning process. This planning predicted possible moves by the opponent and future scenarios based on the current state of the game.
The planning results are provided below as a reference:\n
#Simulated Planning History:\n{planning}\nSimultion ends.\n\n

Now, carefully review the simulated planning history and reflect and decide how many items you will take (between 1 and {initial_max_take}) on this turn.\n

The output should be a markdown code snippet formatted in the following schema, including the leading and trailing \\`\\`\\`json" and "\\`\\`\\`":\n\n```\n{{\n\t"reasoning": string  // This is the reasons for the action\n\t"action": integer  // This is an action you take based on the reasoning. Only provide integer between 1 and {initial_max_take}. You cannot choose 0.\n}}
        """
    # print('final prompt', final_prompt_agent1)
    if agent1["model"] == 'llama':
        while True:
            try:
                system_prompt = "You are a skilled Nim player."
                response = say_model(system_prompt, final_prompt_agent1)
                content = response.split("<|start_header_id|>assistant<|end_header_id|>\n")[1]
                matches_with_braces = re.search(r'\{.*?\}', content, re.DOTALL)
                parsed_content_with_braces = matches_with_braces.group(0) if matches_with_braces else None
                try: 
                    parsed_content_with_braces = parsed_content_with_braces.replace('\xa0', '').strip()       
                except (AttributeError, json.JSONDecodeError) as e:
                    print(f"Error encountered: {e}. Retrying...")
                    continue  # Retry by calling say_model again
                try:
                    parsed_content = json.loads(parsed_content_with_braces)
                except (AttributeError, json.JSONDecodeError) as e:
                    print(f"Error encountered: {e}. Retrying...")
                    continue  # Retry by calling say_model again
                break
            
            except (AttributeError, json.JSONDecodeError) as e:
                print(f"Error encountered: {e}. Retrying...")
                continue  # Retry by calling say_model again

    elif agent1["model"] == 'gemma':
        while True:
            try:
                system_prompt = "You are a skilled Nim player."
                response = say_model(system_prompt, final_prompt_agent1)
                content = response.split("<start_of_turn>model\n")[1]
                matches_with_braces = re.search(r'\{.*?\}', content, re.DOTALL)
                parsed_content_with_braces = matches_with_braces.group(0) if matches_with_braces else None
                try: 
                    parsed_content_with_braces = parsed_content_with_braces.replace('\xa0', '').strip()       
                except (AttributeError, json.JSONDecodeError) as e:
                    print(f"Error encountered: {e}. Retrying...")
                    continue  # Retry by calling say_model again
                try:
                    parsed_content = json.loads(parsed_content_with_braces)
                except (AttributeError, json.JSONDecodeError) as e:
                    print(f"Error encountered: {e}. Retrying...")
                    continue  # Retry by calling say_model again
                break
            
            except (AttributeError, json.JSONDecodeError) as e:
                print(f"Error encountered: {e}. Retrying...")
                continue  # Retry by calling say_model again

    elif agent1["model"] == 'gemini-1.5-flash' or agent1["model"] == 'gemini-1.5-pro':

        # Create the model
        generation_config = {
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 8192,
        "response_mime_type": "application/json",
        }

        model = genai.GenerativeModel(
        model_name=agent1["model"],
        generation_config=generation_config,
        system_instruction="You are a skilled Nim player.",
        )

        chat_session = model.start_chat(
        history=[
        ]
        )

        response = chat_session.send_message(f"{final_prompt_agent1}")

        content = response.text
    
    elif agent1["model"] == 'gpt-4o' or agent1["model"] == 'gpt-4o-mini' or agent1["model"] == 'gpt-3.5-turbo':
        while True:
            try:
                response = client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a skilled Nim player.",
                    },
                    {
                        "role": "user",
                        "content": f"{final_prompt_agent1}",
                    }
                ],
                model=agent1["model"],
                temperature=1
                )
                content = response.choices[0].message.content
                matches_with_braces = re.search(r'\{.*?\}', content, re.DOTALL)
                parsed_content_with_braces = matches_with_braces.group(0) if matches_with_braces else None
                parsed_content_with_braces = parsed_content_with_braces.replace('\xa0', '').strip()

                parsed_content = json.loads(parsed_content_with_braces)
                break
            except (AttributeError, json.JSONDecodeError) as e:
                print(f"Error encountered: {e}. Retrying...")
                continue  # Retry by calling say_model agai

    matches_with_braces = re.search(r'\{.*?\}', content, re.DOTALL)

    parsed_content_with_braces = matches_with_braces.group(0) if matches_with_braces else None
    parsed_content_with_braces = parsed_content_with_braces.replace('\xa0', '').strip()
    parsed_content = json.loads(parsed_content_with_braces)

    agent1_reasoning = parsed_content.get("reasoning")
    agent1_action = parsed_content.get("action")
    # print('My reasoning:', agent1_reasoning)
    # print('My action:', agent1_action)
    # print('################################################################################################################')

    agent1_final_move = int(agent1_action)
    
    return agent1_reasoning, agent1_final_move



def get_move_with_debate(agent1, agent2, remaining_items, max_take, last_taken):
    initial_moves = {}
    initial_reasonings = {}
    i = 0
    for agent in [agent1, agent2]:
        if last_taken is None:
            prompt = f"""
#Game Role:\n You are {agent['name']}, a participant in a simple game.\n\n
#Objective:\n Your goal is to win the game by avoiding taking the last remaining item. The person who takes the last item loses.\n\n
#Game Rule:\n 1. There is a single pile of stones.\n
2. Players take turns to take stones.\n
3. The first player can take any number of stones, but not all the stones in the first move.\n
4. On subsequent turns, the number of stones a player can take must be at least 1 and at most twice the number of stones the previous player took.\n
5. The player who takes the last stone wins the game.\n\n
#Current State:\n There are {remaining_items} stones remaining in the pile.\n
You can take between 1 and {max_take-1} stones on your turn, where {max_take-1} = min(2 × {last_taken}, {remaining_items-1}).\n\n

#Task:\nYou are the first player. Based on the current state of the game, decide how many items you will take (between 1 and {remaining_items-1}) on this turn.\n\n

The output should be a markdown code snippet formatted in the following schema, including the leading and trailing \\`\\`\\`json" and "\\`\\`\\`":\n\n```\n{{\n\t"reasoning": string  // This is the reasons for the action\n\t"action": integer  // This is an action you take based on the reasoning. Only provide integer between 1 and {max_take-1}. You cannot choose 0.\n}}
    """
        else:
            prompt = f"""
#Game Role:\n You are {agent['name']}, a participant in a simple game.\n\n
#Objective:\n Your goal is to win the game by avoiding taking the last remaining item. The person who takes the last item loses.\n\n
#Game Rule:\n 1. There is a single pile of stones.\n
2. Players take turns to take stones.\n
3. The first player can take any number of stones, but not all the stones in the first move.\n
4. On subsequent turns, the number of stones a player can take must be at least 1 and at most twice the number of stones the previous player took.\n
5. The player who takes the last stone wins the game.\n\n
#Current State:\n There are {remaining_items} stones remaining in the pile.\n
The last player took {last_taken} stones.\n
You can take between 1 and {max_take} stones on your turn, where {max_take} = min(2 × {last_taken}, {remaining_items}).\n\n

#Task:\nBased on the current state of the game, decide how many items you will take (between 1 and {max_take}) on this turn.\n\n

The output should be a markdown code snippet formatted in the following schema, including the leading and trailing \\`\\`\\`json" and "\\`\\`\\`":\n\n```\n{{\n\t"reasoning": string  // This is the reasons for the action\n\t"action": integer  // This is an action you take based on the reasoning. Only provide integer between 1 and {max_take}. You cannot choose 0.\n}}
"""

        if agent["model"] == 'llama':
            while True:
                try:
                    system_prompt = "You are a skilled Nim player and debating the best move."
                    response = say_model(system_prompt, prompt)
                    content = response.split("<|start_header_id|>assistant<|end_header_id|>\n")[1]
                    matches_with_braces = re.search(r'\{.*?\}', content, re.DOTALL)
                    parsed_content_with_braces = matches_with_braces.group(0) if matches_with_braces else None
                    try: 
                        parsed_content_with_braces = parsed_content_with_braces.replace('\xa0', '').strip()       
                    except (AttributeError, json.JSONDecodeError) as e:
                        print(f"Error encountered: {e}. Retrying...")
                        continue  # Retry by calling say_model again
                    try:
                        parsed_content = json.loads(parsed_content_with_braces)
                    except (AttributeError, json.JSONDecodeError) as e:
                        print(f"Error encountered: {e}. Retrying...")
                        continue  # Retry by calling say_model again
                    break
                
                except (AttributeError, json.JSONDecodeError) as e:
                    print(f"Error encountered: {e}. Retrying...")
                    continue  # Retry by calling say_model again

        elif agent["model"] == 'gemma':
            while True:
                try:
                    system_prompt = "You are a skilled Nim player and debating the best move."
                    response = say_model(system_prompt, prompt)
                    content = response.split("<start_of_turn>model\n")[1]
                    matches_with_braces = re.search(r'\{.*?\}', content, re.DOTALL)
                    parsed_content_with_braces = matches_with_braces.group(0) if matches_with_braces else None
                    try: 
                        parsed_content_with_braces = parsed_content_with_braces.replace('\xa0', '').strip()       
                    except (AttributeError, json.JSONDecodeError) as e:
                        print(f"Error encountered: {e}. Retrying...")
                        continue  # Retry by calling say_model again
                    try:
                        parsed_content = json.loads(parsed_content_with_braces)
                    except (AttributeError, json.JSONDecodeError) as e:
                        print(f"Error encountered: {e}. Retrying...")
                        continue  # Retry by calling say_model again
                    break
                
                except (AttributeError, json.JSONDecodeError) as e:
                    print(f"Error encountered: {e}. Retrying...")
                    continue  # Retry by calling say_model again

        elif agent["model"] == 'gemini-1.5-flash' or agent["model"] == 'gemini-1.5-pro':

            # Create the model
            generation_config = {
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 8192,
            "response_mime_type": "application/json",
            }

            model = genai.GenerativeModel(
            model_name=agent["model"],
            generation_config=generation_config,
            system_instruction="You are a skilled Nim player and debating the best move.",
            )

            chat_session = model.start_chat(
            history=[
            ]
        )

            response = chat_session.send_message(f"{prompt}")

            content = response.text

        elif agent["model"] == 'gpt-4o' or agent["model"] == 'gpt-4o-mini' or agent["model"] == 'gpt-3.5-turbo':
            while True:
                try:
                    response = client.chat.completions.create(
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a skilled Nim player and debating the best move.",
                        },
                        {
                            "role": "user",
                            "content": f"{prompt}",
                        }
                    ],
                    model=agent["model"],
                    temperature=0.7
                    )
                    content = response.choices[0].message.content
                    matches_with_braces = re.search(r'\{.*?\}', content, re.DOTALL)
                    parsed_content_with_braces = matches_with_braces.group(0) if matches_with_braces else None
                    parsed_content_with_braces = parsed_content_with_braces.replace('\xa0', '').strip()

                    parsed_content = json.loads(parsed_content_with_braces)
                    break
                except (AttributeError, json.JSONDecodeError) as e:
                    print(f"Error encountered: {e}. Retrying...")
                    continue  # Retry by calling say_model again

        matches_with_braces = re.search(r'\{.*?\}', content, re.DOTALL)

        parsed_content_with_braces = matches_with_braces.group(0) if matches_with_braces else None
        parsed_content_with_braces = parsed_content_with_braces.replace('\xa0', '').strip()
        parsed_content = json.loads(parsed_content_with_braces)

        initial_reasoning = parsed_content.get("reasoning")
        initial_action = parsed_content.get("action")
        if i == 0:
            initial_moves['agent1'] = initial_action
            initial_reasonings['agent1'] = initial_reasoning
        if i == 1:
            initial_moves['agent2'] = initial_action
            initial_reasonings['agent2'] = initial_reasoning
        i += 1
        # print('initial reasoning', initial_reasonings)
        # print('initial action', initial_moves)

        
    # If agents agree, return move
    # if len(set(initial_moves.values())) == 1:
    #     return initial_reasoning, list(initial_moves.values())[0]

    # Otherwise, conduct debate rounds to reach consensus
    for _ in range(debate_rounds):
        # agent_moves_str = "\n".join(f"{name}: {move}" for name, move in initial_moves.items())
        # refined_moves = {}
        i = 0
        for agent in [agent1, agent2]:
            # others = [a for a in [agent1, agent2] if a != agent]
            # other = others[0]
            if i == 0:
                if last_taken is None:
                    prompt = f"""
#Game Role:\n You are {agent['name']}, a participant in a simple game.\n\n
#Objective:\n Your goal is to win the game by avoiding taking the last remaining item. The person who takes the last item loses.\n\n
#Game Rule:\n 1. There is a single pile of stones.\n
2. Players take turns to take stones.\n
3. The first player can take any number of stones, but not all the stones in the first move.\n
4. On subsequent turns, the number of stones a player can take must be at least 1 and at most twice the number of stones the previous player took.\n
5. The player who takes the last stone wins the game.\n\n
#Current State:\n There are {remaining_items} stones remaining in the pile.\n
You can take between 1 and {max_take-1} stones on your turn, where {max_take-1} = min(2 × {last_taken}, {remaining_items-1}).\n\n

#Task:\nYou are the first player. Based on the current state of the game, decide how many items you will take (between 1 and {remaining_items-1}) on this turn.\n\n

You initially chose {initial_moves['agent1']} items at first trial by the reason: '{initial_reasonings['agent1']}'.\n
Other agent argues that you have to choose move as: {initial_moves['agent2']} by the reason: {initial_reasonings['agent2']}.\n
Considering the other's opinion, refine or confirm your move.\n

The output should be a markdown code snippet formatted in the following schema, including the leading and trailing \\`\\`\\`json" and "\\`\\`\\`":\n\n```\n{{\n\t"reasoning": string  // This is the reasons for the action\n\t"action": integer  // This is an action you take based on the reasoning. Only provide integer between 1 and {max_take-1}. You cannot choose 0.\n}}
"""
                else:
                    prompt = f"""
#Game Role:\n You are {agent['name']}, a participant in a simple game.\n\n
#Objective:\n Your goal is to win the game by avoiding taking the last remaining item. The person who takes the last item loses.\n\n
#Game Rule:\n 1. There is a single pile of stones.\n
2. Players take turns to take stones.\n
3. The first player can take any number of stones, but not all the stones in the first move.\n
4. On subsequent turns, the number of stones a player can take must be at least 1 and at most twice the number of stones the previous player took.\n
5. The player who takes the last stone wins the game.\n\n
#Current State:\n There are {remaining_items} stones remaining in the pile.\n
The last player took {last_taken} stones.\n
You can take between 1 and {max_take} stones on your turn, where {max_take} = min(2 × {last_taken}, {remaining_items}).\n\n

#Task:\nBased on the current state of the game, decide how many items you will take (between 1 and {max_take}) on this turn.\n\n

You initially chose {initial_moves['agent1']} items at first trial by the reason: '{initial_reasonings['agent1']}'.\n
Other agent argues that you have to choose move as: {initial_moves['agent2']} by the reason: {initial_reasonings['agent2']}.\n
Considering the other's opinion, refine or confirm your move.\n

The output should be a markdown code snippet formatted in the following schema, including the leading and trailing \\`\\`\\`json" and "\\`\\`\\`":\n\n```\n{{\n\t"reasoning": string  // This is the reasons for the action\n\t"action": integer  // This is an action you take based on the reasoning. Only provide integer between 1 and {max_take}. You cannot choose 0.\n}}
"""
            if i == 1:
                if last_taken is None:
                    prompt = f"""
#Game Role:\n You are {agent['name']}, a participant in a simple game.\n\n
#Objective:\n Your goal is to win the game by avoiding taking the last remaining item. The person who takes the last item loses.\n\n
#Game Rule:\n 1. There is a single pile of stones.\n
2. Players take turns to take stones.\n
3. The first player can take any number of stones, but not all the stones in the first move.\n
4. On subsequent turns, the number of stones a player can take must be at least 1 and at most twice the number of stones the previous player took.\n
5. The player who takes the last stone wins the game.\n\n
#Current State:\n There are {remaining_items} stones remaining in the pile.\n
You can take between 1 and {max_take-1} stones on your turn, where {max_take-1} = min(2 × {last_taken}, {remaining_items-1}).\n\n

#Task:\nYou are the first player. Based on the current state of the game, decide how many items you will take (between 1 and {remaining_items-1}) on this turn.\n\n

You initially chose {initial_moves['agent2']} items at first trial by the reason: '{initial_reasonings['agent2']}'.\n
Other agent argues that you have to choose move as: {initial_moves['agent1']} by the reason: {initial_reasonings['agent1']}.\n
Considering the other's opinion, refine or confirm your move.\n

The output should be a markdown code snippet formatted in the following schema, including the leading and trailing \\`\\`\\`json" and "\\`\\`\\`":\n\n```\n{{\n\t"reasoning": string  // This is the reasons for the action\n\t"action": integer  // This is an action you take based on the reasoning. Only provide integer between 1 and {max_take-1}. You cannot choose 0.\n}}
"""
                else:
                    prompt = f"""
#Game Role:\n You are {agent['name']}, a participant in a simple game.\n\n
#Objective:\n Your goal is to win the game by avoiding taking the last remaining item. The person who takes the last item loses.\n\n
#Game Rule:\n 1. There is a single pile of stones.\n
2. Players take turns to take stones.\n
3. The first player can take any number of stones, but not all the stones in the first move.\n
4. On subsequent turns, the number of stones a player can take must be at least 1 and at most twice the number of stones the previous player took.\n
5. The player who takes the last stone wins the game.\n\n
#Current State:\n There are {remaining_items} stones remaining in the pile.\n
The last player took {last_taken} stones.\n
You can take between 1 and {max_take} stones on your turn, where {max_take} = min(2 × {last_taken}, {remaining_items}).\n\n

#Task:\nBased on the current state of the game, decide how many items you will take (between 1 and {max_take}) on this turn.\n\n

You initially chose {initial_moves['agent2']} items at first trial by the reason: '{initial_reasonings['agent2']}'.\n
Other agent argues that you have to choose move as: {initial_moves['agent1']} by the reason: {initial_reasonings['agent1']}.\n
Considering the other's opinion, refine or confirm your move.\n

The output should be a markdown code snippet formatted in the following schema, including the leading and trailing \\`\\`\\`json" and "\\`\\`\\`":\n\n```\n{{\n\t"reasoning": string  // This is the reasons for the action\n\t"action": integer  // This is an action you take based on the reasoning. Only provide integer between 1 and {max_take}. You cannot choose 0.\n}}
"""

            if agent["model"] == 'llama':
                while True:
                    try:
                        system_prompt = "You are a skilled Nim player and debating the best move."
                        response = say_model(system_prompt, prompt)
                        content = response.split("<|start_header_id|>assistant<|end_header_id|>\n")[1]
                        matches_with_braces = re.search(r'\{.*?\}', content, re.DOTALL)
                        parsed_content_with_braces = matches_with_braces.group(0) if matches_with_braces else None
                        try: 
                            parsed_content_with_braces = parsed_content_with_braces.replace('\xa0', '').strip()       
                        except (AttributeError, json.JSONDecodeError) as e:
                            print(f"Error encountered: {e}. Retrying...")
                            continue  # Retry by calling say_model again
                        try:
                            parsed_content = json.loads(parsed_content_with_braces)
                        except (AttributeError, json.JSONDecodeError) as e:
                            print(f"Error encountered: {e}. Retrying...")
                            continue  # Retry by calling say_model again
                        break
                    
                    except (AttributeError, json.JSONDecodeError) as e:
                        print(f"Error encountered: {e}. Retrying...")
                        continue  # Retry by calling say_model again

            elif agent["model"] == 'gemma':
                while True:
                    try:
                        system_prompt = "You are a skilled Nim player and debating the best move."
                        response = say_model(system_prompt, prompt)
                        content = response.split("<start_of_turn>model\n")[1]
                        matches_with_braces = re.search(r'\{.*?\}', content, re.DOTALL)
                        parsed_content_with_braces = matches_with_braces.group(0) if matches_with_braces else None
                        try: 
                            parsed_content_with_braces = parsed_content_with_braces.replace('\xa0', '').strip()       
                        except (AttributeError, json.JSONDecodeError) as e:
                            print(f"Error encountered: {e}. Retrying...")
                            continue  # Retry by calling say_model again
                        try:
                            parsed_content = json.loads(parsed_content_with_braces)
                        except (AttributeError, json.JSONDecodeError) as e:
                            print(f"Error encountered: {e}. Retrying...")
                            continue  # Retry by calling say_model again
                        break
                    
                    except (AttributeError, json.JSONDecodeError) as e:
                        print(f"Error encountered: {e}. Retrying...")
                        continue  # Retry by calling say_model again

            elif agent["model"] == 'gemini-1.5-flash' or agent["model"] == 'gemini-1.5-pro':

                # Create the model
                generation_config = {
                "temperature": 0.7,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 8192,
                "response_mime_type": "application/json",
                }

                model = genai.GenerativeModel(
                model_name=agent["model"],
                generation_config=generation_config,
                system_instruction="You are a skilled Nim player and debating the best move.",
                )

                chat_session = model.start_chat(
                history=[
                ]
            )

                response = chat_session.send_message(f"{prompt}")

                content = response.text
            
            elif agent["model"] == 'gpt-4o' or agent["model"] == 'gpt-4o-mini' or agent["model"] == 'gpt-3.5-turbo':
                while True:
                    try:
                        response = client.chat.completions.create(
                        messages=[
                            {
                                "role": "system",
                                "content": "You are a skilled Nim player and debating the best move.",
                            },
                            {
                                "role": "user",
                                "content": f"{prompt}",
                            }
                        ],
                        model=agent["model"],
                        temperature=0.7
                        )
                        content = response.choices[0].message.content
                        matches_with_braces = re.search(r'\{.*?\}', content, re.DOTALL)
                        parsed_content_with_braces = matches_with_braces.group(0) if matches_with_braces else None
                        parsed_content_with_braces = parsed_content_with_braces.replace('\xa0', '').strip()

                        parsed_content = json.loads(parsed_content_with_braces)
                        break
                    except (AttributeError, json.JSONDecodeError) as e:
                        print(f"Error encountered: {e}. Retrying...")
                        continue  # Retry by calling say_model again

            matches_with_braces = re.search(r'\{.*?\}', content, re.DOTALL)

            parsed_content_with_braces = matches_with_braces.group(0) if matches_with_braces else None
            parsed_content_with_braces = parsed_content_with_braces.replace('\xa0', '').strip()
            parsed_content = json.loads(parsed_content_with_braces)

            initial_reasoning = parsed_content.get("reasoning")
            initial_action = parsed_content.get("action")
            if i == 0:
                initial_moves['agent1'] = initial_action
                initial_reasonings['agent1'] = initial_reasoning
            if i == 1:
                initial_moves['agent2'] = initial_action
                initial_reasonings['agent2'] = initial_reasoning

            i += 1
            # initial_moves[agent["name"]] = initial_action
            # initial_reasonings[agent["name"]] = initial_reasoning

        # initial_moves = refined_moves
        if len(set(initial_moves.values())) == 1:
            return initial_reasoning, initial_action

    return initial_reasoning, Counter(initial_moves.values()).most_common(1)[0][0]  # Use most common if no consensus

def play_fibonacci_nim_game(total_items, verbose=False):
    # Create output file path
    file_path = f'/home/jihwan/NashIP/result/FN20_M/{args.agent1_model}_{args.agent1_prompt}_{n_step_lookahead}_{args.agent2_model}_{args.agent2_prompt}.txt'
    with open(file_path, 'a') as f:
        current_items = total_items
        turn = 0
        last_taken = None  # Tracks the number of stones taken in the previous turn

        while current_items > 0:
            # Determine the current and other agents
            current_agent = agents[turn % 2]
            other_agent = agents[(turn + 1) % 2]

            # Determine the max_take for the current turn
            max_take = current_items if last_taken is None else min(2 * last_taken, current_items)

            # Decide move based on the agent's prompting method
            if current_agent["prompting_method"] == "self_consistency":
                reasoning, move = get_consistent_move(current_agent, current_items, self_consistency_count, max_take, last_taken)
            # elif current_agent["prompting_method"] == "n_step_lookahead":
            #     move = get_move_with_n_step_lookahead(current_agent, other_agent, current_items)
            elif current_agent["prompting_method"] == "self_reflection":
                reasoning, move = get_move_with_reflection(current_agent, current_items, max_take, last_taken)
            elif current_agent["prompting_method"] == "debate":
                reasoning, move = get_move_with_debate(current_agent, current_agent, current_items, max_take, last_taken)
            elif current_agent["prompting_method"] == "self_play_debate":
                reasoning, move = self_play_debate(current_agent, other_agent, current_items, n_step_lookahead, max_take, last_taken)
            elif current_agent["prompting_method"] == "self_play_debate_exp":
                reasoning, move = self_play_debate_exp(current_agent, other_agent, current_items, n_step_lookahead, max_take, last_taken)
            elif current_agent["prompting_method"] == "basic":
                reasoning, move = get_basic_move(current_agent, current_items, max_take, last_taken)
            else:
                print("Error: set the prompting methods")
                return None

            # Output reasoning and move
            if verbose:
                print(f"Reasoning: {reasoning}\nAction: {move}", file=f)
                print(f"{current_agent['name']} ({current_agent['model']}) takes {move} items. Items remaining: {current_items - move}", file=f)

            # Update the game state
            current_items -= move
            last_taken = move  # Update the last taken value

            # Check for win condition
            if current_items <= 0:
                if verbose:
                    print(f"{other_agent['name']} ({other_agent['model']}) wins!", file=f)
                return other_agent["name"]

            # Move to the next turn
            turn += 1

def simulate_fibonacci_nim_games(num_games, total_items):
    # Initialize win counts for agents
    win_counts = {agent["name"]: 0 for agent in agents}
    file_path = f'/home/jihwan/NashIP/result/FN20_M/{args.agent1_model}_{args.agent1_prompt}_{n_step_lookahead}_{args.agent2_model}_{args.agent2_prompt}.txt'

    # Run the simulation for the specified number of games
    with open(file_path, 'a') as f:
        for game_num in range(num_games):
            print(f"\nStarting Game {game_num + 1}", file=f)
            print(f"\nStarting Game {game_num + 1}")
            winner = play_fibonacci_nim_game(total_items, verbose=True)
            win_counts[winner] += 1

        # Output game results
        print("\nGame Results:", file=f)
        for agent in agents:
            win_rate = (win_counts[agent["name"]] / num_games) * 100
            print(f"{agent['name']} Win Rate: {win_rate:.2f}% ({win_counts[agent['name']]} wins out of {num_games})", file=f)



simulate_fibonacci_nim_games(num_games, total_items)