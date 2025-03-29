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

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4, 5'

import os
import google.generativeai as genai
genai.configure(api_key='AIzaSyCYkix3fQio-WpUus7ziwNGhSk6qZp7LJs')

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"),)



parser = argparse.ArgumentParser(description='arguments for training')

parser.add_argument('--agent1_model',     type=str,   default=None, help='model')
parser.add_argument('--agent2_model',     type=str,   default=None, help='model')
parser.add_argument('--agent1_prompt',     type=str,   default='basic', help='prompt_method')
parser.add_argument('--agent2_prompt',     type=str,   default='basic', help='prompt_method')
parser.add_argument('--num_games',     type=int,   default='50', help='prompt_method')
parser.add_argument('--look_ahead',     type=int,   default='0', help='prompt_method')
parser.add_argument('--temperature',     type=float,   default='0.7', help='prompt_method')

args = parser.parse_args()

# Set up your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize the game parameters
total_items = 31  # Total items in the pile (e.g., 21)
max_take = 3  # Maximum items that can be taken per turn
num_games = args.num_games  # Number of games to play
num_refine = 3
self_consistency_count = 10  # Number of responses to use for self-consistency
n_step_lookahead = args.look_ahead  # Number of lookahead steps for n-step opponent modeling
debate_rounds = 3  # Maximum number of debate rounds
spc_temperature = args.temperature

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
    print("Small Language Models: ", args.agent1_model)
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
        if args.agent1_model == 'llama':
            inputs = tokenizer(convert_to_llama_format(system, instruction_str), return_tensors = "pt").to("cuda")
            if args.agent1_prompt == 'self_consistency':
                outputs = model.generate(**inputs, max_new_tokens = 500, use_cache = True, temperature = 0.7, top_p = 0.95, pad_token_id = tokenizer.eos_token_id)
            else:
                outputs = model.generate(**inputs, max_new_tokens = 500, use_cache = True, temperature = 0.7, top_p = 0.95, pad_token_id = tokenizer.eos_token_id)
            return(tokenizer.batch_decode(outputs)[0])
        
        elif args.agent1_model == 'gemma':
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

        elif args.agent1_model == 'qwen':
            messages = [
            {"role": "system", "content": f"{system}"},
            {"role": "user", "content": f"{instruction_str}"}]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
            generated_ids = model.generate(**inputs, max_new_tokens=500, temperature = 0.7, top_p = 0.95,)
            generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)]
            return(tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0])
        
        elif args.agent1_model == 'mistral':
            messages = [{"role": "user", "content": f"{system}\n{instruction_str}"},]
            prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_dict=True, return_tensors="pt")
            prompt.to("cuda")
            outputs = model.generate(**prompt, max_new_tokens=500, temperature = 0.7, top_p = 0.95,)
            # inputs = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to("cuda")
            return(tokenizer.batch_decode(outputs[0], skip_special_tokens=True))

def get_agent_response(agent, prompt, system_prompt="You are a skilled Nim player.", temperature = 0.7):
    while True:
        try:
            if agent["model"] in ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-1.0-pro"]:
                generation_config = {
                    "temperature": temperature,
                    "top_p": 0.95,
                    "top_k": 40,
                    "max_output_tokens": 8192,
                    "response_mime_type": "application/json",
                }
                model = genai.GenerativeModel(
                    model_name=agent["model"],
                    generation_config=generation_config,
                    system_instruction=system_prompt,
                )
                chat_session = model.start_chat(history=[])
                response = chat_session.send_message(prompt)
                try:
                    return json.loads(response.text)
                except json.JSONDecodeError:
                    print("Error decoding JSON for gemini model. Retrying...")

            elif agent["model"] in ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"]:
                response = client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    model=agent["model"],
                    temperature=temperature,
                )
                content = response.choices[0].message.content
                parsed_content = json.loads(re.search(r'\{.*?\}', content, re.DOTALL).group(0).replace('\xa0', '').strip())
                if parsed_content:
                    return parsed_content

            print("Error encountered. Retrying in 2 seconds...")
            time.sleep(2)  # Delay to prevent rapid retries
        except KeyboardInterrupt:
            print("Process interrupted by user.")
            return None
        except Exception as e:
            print(f"Unexpected error: {e}. Retrying...")

# def get_agent_response(agent, prompt, system_prompt="You are a skilled Nim player."): #llama, gemma, qwen, mistral 
#     """
#     Handles responses for different models based on the agent's configuration.

#     Parameters:
#         agent (dict): Dictionary containing agent details, including the model name.
#         prompt (str): The input prompt for the model.
#         system_prompt (str): The system-level instruction for the model.

#     Returns:
#         dict: Parsed content from the model's response in JSON format, or None if parsing fails.
#     """
#     def parse_content(response, split_key):
#         try:
#             if split_key not in response:
#                 return None
#             content = response.split(split_key)[1]
#             matches_with_braces = re.search(r'\{.*?\}', content, re.DOTALL)
#             parsed_content_with_braces = matches_with_braces.group(0).replace('\xa0', '').strip() if matches_with_braces else None
#             return json.loads(parsed_content_with_braces) if parsed_content_with_braces else None
#         except (AttributeError, json.JSONDecodeError, IndexError):
#             return None

#     while True:
#         try:
#             if agent["model"] == "llama":
#                 split_key = "<|start_header_id|>assistant<|end_header_id|>\n"
#                 response = say_model(system_prompt, prompt)
#                 parsed_content = parse_content(response, split_key)
#                 if parsed_content:
#                     return parsed_content

#             elif agent["model"] == "gemma":
#                 split_key = "<start_of_turn>model\n"
#                 response = say_model(system_prompt, prompt)
#                 parsed_content = parse_content(response, split_key)
#                 if parsed_content:
#                     return parsed_content

#             elif agent["model"] in ["gemini-1.5-flash", "gemini-1.5-pro"]:
#                 generation_config = {
#                     "temperature": 0.7,
#                     "top_p": 0.95,
#                     "top_k": 40,
#                     "max_output_tokens": 8192,
#                     "response_mime_type": "application/json",
#                 }
#                 model = genai.GenerativeModel(
#                     model_name=agent["model"],
#                     generation_config=generation_config,
#                     system_instruction=system_prompt,
#                 )
#                 chat_session = model.start_chat(history=[])
#                 response = chat_session.send_message(prompt)
#                 try:
#                     return json.loads(response.text)
#                 except json.JSONDecodeError:
#                     print("Error decoding JSON for gemini model. Retrying...")

#             elif agent["model"] in ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"]:
#                 response = client.chat.completions.create(
#                     messages=[
#                         {"role": "system", "content": system_prompt},
#                         {"role": "user", "content": prompt},
#                     ],
#                     model=agent["model"],
#                     temperature=0.7,
#                 )
#                 content = response.choices[0].message.content
#                 parsed_content = json.loads(re.search(r'\{.*?\}', content, re.DOTALL).group(0).replace('\xa0', '').strip())
#                 if parsed_content:
#                     return parsed_content

#             print("Error encountered. Retrying in 2 seconds...")
#             time.sleep(2)  # Delay to prevent rapid retries
#         except KeyboardInterrupt:
#             print("Process interrupted by user.")
#             return None
#         except Exception as e:
#             print(f"Unexpected error: {e}. Retrying...")


agents = [
    {"name": "Agent 1", "model": args.agent1_model, "prompting_method": args.agent1_prompt},
    {"name": "Agent 2", "model": args.agent2_model, "prompting_method": args.agent2_prompt}
]

# Function for basic move (single-response without consistency or modeling)
def get_move(agent, remaining_items):
    json_file_path = f'/home/jihwan/NashIP/result/BR31/{args.agent1_model}_{args.agent1_prompt}_{n_step_lookahead}_{args.agent2_model}_{args.agent2_prompt}.json'
    Path(json_file_path).parent.mkdir(parents=True, exist_ok=True)
    # with open(f'/home/jihwan/NashIP/result/BR31/{args.agent1_model}_{args.agent1_prompt}_{n_step_lookahead}_{args.agent2_model}_{args.agent2_prompt}.txt', 'a') as f:
    prompt = f"""
    #Game Role:\n You are {agent['name']}, a participant in a game of Nim variants.\n\n
    #Objective:\n Your goal is to win the game by taking all remaining items on your turn, leaving no items for your opponent. The person who takes the last item wins.\n\n
    #Game Rule:\n There is a single pile of items. You can take between 1 and {max_take} items on your turn.\n\n
    #Current State:\n There are {remaining_items} items remaining in the pile.\n\n
    #Task:\nBased on the current state of the game, decide how many items you will take (between 1 and {max_take}) on this turn.\n\n

    The output should be a markdown code snippet formatted in the following schema, including the leading and trailing \\`\\`\\`json" and "\\`\\`\\`":\n\n```\n{{\n\t"action": integer  // This is an action you take based on the reasoning. Only provide integer between 1 and 3.\n}}
    """

    parsed_content = get_agent_response(agent, prompt, system_prompt="You are a skilled Nim player.")

    action = parsed_content.get("action")
    if int(action) > 3:
        action = 3
    if int(action) < 1:
        action = 1

    return "None", action

def get_basic_move(agent, remaining_items):
    json_file_path = f'/home/jihwan/NashIP/result/BR31/{args.agent1_model}_{args.agent1_prompt}_{n_step_lookahead}_{args.agent2_model}_{args.agent2_prompt}.json'
    Path(json_file_path).parent.mkdir(parents=True, exist_ok=True)
    # with open(f'/home/jihwan/NashIP/result/BR31/{args.agent1_model}_{args.agent1_prompt}_{n_step_lookahead}_{args.agent2_model}_{args.agent2_prompt}.txt', 'a') as f:
    prompt = f"""
    #Game Role:\n You are {agent['name']}, a participant in a game of Nim variants.\n\n
    #Objective:\n Your goal is to win the game by taking all remaining items on your turn, leaving no items for your opponent. The person who takes the last item wins.\n\n
    #Game Rule:\n There is a single pile of items. You can take between 1 and {max_take} items on your turn.\n\n
    #Current State:\n There are {remaining_items} items remaining in the pile.\n\n
    #Task:\nBased on the current state of the game, decide how many items you will take (between 1 and {max_take}) on this turn.\n\n

    The output should be a markdown code snippet formatted in the following schema, including the leading and trailing \\`\\`\\`json" and "\\`\\`\\`":\n\n```\n{{\n\t"reasoning": string  // This is the reasons for the action\n\t"action": integer  // This is an action you take based on the reasoning. Only provide integer between 1 and 3.\n}}
    """

    parsed_content = get_agent_response(agent, prompt, system_prompt="You are a skilled Nim player.")

    reasoning = parsed_content.get("reasoning")
    action = parsed_content.get("action")
    if int(action) > 3:
        action = 3
    if int(action) < 1:
        action = 1

    return reasoning, action

# Function for self-consistency: generate multiple responses and choose the most common move
def get_consistent_move(agent, remaining_items, num_responses):
    prompt = f"""
    #Game Role:\n You are {agent['name']}, a participant in a game of Nim variants.\n\n
    #Objective:\n Your goal is to win the game by taking all remaining items on your turn, leaving no items for your opponent. The person who takes the last item wins.\n\n
    #Game Rule:\n There is a single pile of items. You can take between 1 and {max_take} items on your turn.\n\n
    #Current State:\n There are {remaining_items} items remaining in the pile.\n\n
    #Task:\nBased on the current state of the game, decide how many items you will take (between 1 and {max_take}) on this turn.\n\n

    The output should be a markdown code snippet formatted in the following schema, including the leading and trailing \\`\\`\\`json" and "\\`\\`\\`":\n\n```\n{{\n\t"reasoning": string  // This is the reasons for the action\n\t"action": integer  // This is an action you take based on the reasoning. Only provide integer between 1 and 3.\n}}
    """
    moves = []

    for _ in range(num_responses):
        parsed_content = get_agent_response(agent, prompt, system_prompt="You are a skilled Nim player.", temperature=1.0)
        reasoning = parsed_content.get("reasoning")
        action = parsed_content.get("action")

        if int(action) > 3:
            action = 3
        if int(action) < 1:
            action = 1
        move = int(action)
        moves.append(move)
    
    most_common_move = Counter(moves).most_common(1)[0][0]

    return reasoning, most_common_move

def get_consistent_diverse_move(agent, remaining_items, num_responses):
    moves = []

    prompt = f"""
        #Game Role:\n You are {agent['name']}, a participant in a game of Nim variants.\n\n
        #Objective:\n Your goal is to win the game by taking all remaining items on your turn, leaving no items for your opponent. The person who takes the last item wins.\n\n
        #Game Rule:\n There is a single pile of items. You can take between 1 and {max_take} items on your turn.\n\n
        #Current State:\n There are {remaining_items} items remaining in the pile.\n\n
        #Task:\nBased on the current state of the game, decide how many items you will take (between 1 and {max_take}) on this turn.\n
        """
        #################prompt1#################
    game_prompt = f"""
    Below is a game description. Extract key information.

    **Game Description:**
    {prompt}

    ### Format Response as:
    {{
    "game_type": "string", // Name of the game (if identifiable).
    "winning_condition": "string", // How to win the game.
    "move_constraints": "string" // What actions are allowed per turn.
    }}
    """

    parsed_content = get_agent_response(agent, game_prompt, system_prompt="You are a game theorist and strategist.",temperature=0.1)

    game_type = parsed_content.get("game_type")
    winning_condition = parsed_content.get("winning_condition")
    move_constraints = parsed_content.get("move_constraints")


    strategy_prompt = f"""
    Based on the game information below, derive the **optimal strategy**.

    **Game:** {game_type}  
    **Winning Condition:** {winning_condition}  
    **Move Constraints:** {move_constraints}

    ### Format Response as:
    {{
    "state_evaluation": "string", // How to assess the game state.
    "winning_strategy": "string", // Winning strategy in this turn to win this game.
    "endgame_tactics": "string" // Best strategy in a near-win situation.}}
    """
    # print("Strategy Prompt: ", strategy_prompt)
    parsed_content = get_agent_response(agent, strategy_prompt, system_prompt="You are a game theorist and strategist.", temperature=0.1)
    # print(2)

    state_evaluation = parsed_content.get("state_evaluation")
    winning_strategy = parsed_content.get("winning_strategy")
    endgame_tactics = parsed_content.get("endgame_tactics")

    for _ in range(num_responses):

        final_prompt = f"""
        Refine the initial game prompt to improve decision-making based on the Game and Strategy.
        ##Initial prompt: {prompt}\n

        **Game:** {game_type}  
        **Strategy:**  
        - State Evaluation: {state_evaluation}  
        - Winning Strategy: {winning_strategy}  
        - Endgame Tactics: {endgame_tactics}  

        ### Instructions:
        1. The new prompt must **clearly guide decision-making**.
        2. It should **force the model to prioritize winning moves**.
        3. Language should be **direct, logical, and assertive**.
        4. Do NOT include the answer—only refine the prompt.
        5. Do NOT define the format of the output.

        ### Format Response as:

        {{
        "optimized_prompt": "string", // The refined prompt that clearly directs decision-making. }}
        """
        parsed_content = get_agent_response(agent, final_prompt, system_prompt="You are a game theorist and strategist.", temperature=spc_temperature)

        optimized_prompt = parsed_content.get("optimized_prompt")

        one_new_prompt = f"""{optimized_prompt}\n

        **Current State:**  
        - There are {remaining_items} items left.  
        - You can take 1 to 3 items per turn. 
        ### Instructions:
        1. **If a winning move exists, take it immediately.**  
        2. **Otherwise, follow optimal move principles.**  
        3. Justify your move using the extracted strategy.

        ### Format Response as:
        {{
        "reasoning": "string", // Explanation of the move based on the strategy.
        "action": integer // Chosen move (1, 2, or 3). }}
        """
        parsed_content = get_agent_response(agent, one_new_prompt, system_prompt="You are a skilled Nim player.", temperature=1.0)
        reasoning = parsed_content.get("reasoning")
        action = parsed_content.get("action")

        if int(action) > 3:
            action = 3
        if int(action) < 1:
            action = 1
        move = int(action)
        moves.append(move)
    
    most_common_move = Counter(moves).most_common(1)[0][0]

    return reasoning, most_common_move

# Function for self-reflection prompting
def get_move_with_reflection(agent, remaining_items):
    prompt_initial = f"""
    #Game Role:\n You are {agent['name']}, a participant in a game of Nim variants.\n\n
    #Objective:\n Your goal is to win the game by taking all remaining items on your turn, leaving no items for your opponent. The person who takes the last item wins.\n\n
    #Game Rule:\n There is a single pile of items. You can take between 1 and {max_take} items on your turn.\n\n
    #Current State:\n There are {remaining_items} items remaining in the pile.\n\n
    #Task:\nBased on the current state of the game, decide how many items you will take (between 1 and {max_take}) on this turn.\n\n

    The output should be a markdown code snippet formatted in the following schema, including the leading and trailing \\`\\`\\`json" and "\\`\\`\\`":\n\n```\n{{\n\t"reasoning": string  // This is the reasons for the action\n\t"action": integer  // This is an action you take based on the reasoning. Only provide integer between 1 and 3.\n}}
    """

    parsed_content = get_agent_response(agent, prompt_initial, system_prompt="You are a skilled Nim player.")

    initial_reasoning = parsed_content.get("reasoning")
    action = parsed_content.get("action")

    initial_move = int(action)

    for k in range(num_refine):

        feedback_prompt = f"""
        #Game Role:\n You are {agent['name']}, a participant in a game of Nim variants.\n\n
        #Objective:\n Your goal is to win the game by taking all remaining items on your turn, leaving no items for your opponent. The person who takes the last item wins.\n\n
        #Game Rule:\n There is a single pile of items. You can take between 1 and {max_take} items on your turn.\n\n
        #Current State:\n There are {remaining_items} items remaining in the pile.\n\n
        #Task:\nBased on the current state of the game, give a feedback on the first trial's reasoning and action.\n\n
        #First trial's reasoning and action:\nYou initially chose {initial_move} items at first trial by the reason: '{initial_reasoning}'.\n\n

        The output should be a markdown code snippet formatted in the following schema, including the leading and trailing \\`\\`\\`json" and "\\`\\`\\`":\n\n```\n{{\n\t"feedback": string  // This is the feedback for the selected action and reasoning\n}}
        """

        parsed_content = get_agent_response(agent, feedback_prompt, system_prompt="You are a skilled Nim player.")
        feedback = parsed_content.get("feedback")

        refine_prompt = f"""
        #Game Role:\n You are {agent['name']}, a participant in a game of Nim variants.\n\n
        #Objective:\n Your goal is to win the game by taking all remaining items on your turn, leaving no items for your opponent. The person who takes the last item wins.\n\n
        #Game Rule:\n There is a single pile of items. You can take between 1 and {max_take} items on your turn.\n\n
        #Current State:\n There are {remaining_items} items remaining in the pile.\n\n

        You initially chose {initial_move} items at first trial by the reason: '{initial_reasoning}'.\n\n
        You recieved feedback on your action and reasoning: {feedback}\n\n
        #Task:\nBased on the current state of the game and the feedback, refine your reasoning and action. And finally, decide how many items you will take (between 1 and {max_take}) on this turn.\n\n

        The output should be a markdown code snippet formatted in the following schema, including the leading and trailing \\`\\`\\`json" and "\\`\\`\\`":\n\n```\n{{\n\t"reasoning": string  // This is the reasons for the action\n\t"action": integer  // This is an action you take based on the reasoning. Only provide integer between 1 and 3.\n}}
        """

        parsed_content = get_agent_response(agent, refine_prompt, system_prompt="You are a skilled Nim player.")
        refined_reasoning = parsed_content.get("reasoning")
        refined_action = parsed_content.get("action")

        if initial_move == int(refined_action):
            return refined_reasoning, refined_action
        else:
            initial_move = refined_action
            initial_reasoning = refined_reasoning

    return refined_reasoning, refined_action

def get_move_with_diverse_reflection(agent, remaining_items):
    prompt = f"""
        #Game Role:\n You are {agent['name']}, a participant in a game of Nim variants.\n\n
        #Objective:\n Your goal is to win the game by taking all remaining items on your turn, leaving no items for your opponent. The person who takes the last item wins.\n\n
        #Game Rule:\n There is a single pile of items. You can take between 1 and {max_take} items on your turn.\n\n
        #Current State:\n There are {remaining_items} items remaining in the pile.\n\n
        #Task:\nBased on the current state of the game, decide how many items you will take (between 1 and {max_take}) on this turn.\n
        """
        #################prompt1#################
    game_prompt = f"""
    Below is a game description. Extract key information.

    **Game Description:**
    {prompt}

    ### Format Response as:
    {{
    "game_type": "string", // Name of the game (if identifiable).
    "winning_condition": "string", // How to win the game.
    "move_constraints": "string" // What actions are allowed per turn.
    }}
    """

    parsed_content = get_agent_response(agent, game_prompt, system_prompt="You are a game theorist and strategist.",temperature=0.1)

    game_type = parsed_content.get("game_type")
    winning_condition = parsed_content.get("winning_condition")
    move_constraints = parsed_content.get("move_constraints")


    strategy_prompt = f"""
    Based on the game information below, derive the **optimal strategy**.

    **Game:** {game_type}  
    **Winning Condition:** {winning_condition}  
    **Move Constraints:** {move_constraints}

    ### Format Response as:
    {{
    "state_evaluation": "string", // How to assess the game state.
    "winning_strategy": "string", // Winning strategy in this turn to win this game.
    "endgame_tactics": "string" // Best strategy in a near-win situation.}}
    """
    # print("Strategy Prompt: ", strategy_prompt)
    parsed_content = get_agent_response(agent, strategy_prompt, system_prompt="You are a game theorist and strategist.", temperature=0.1)
    # print(2)

    state_evaluation = parsed_content.get("state_evaluation")
    winning_strategy = parsed_content.get("winning_strategy")
    endgame_tactics = parsed_content.get("endgame_tactics")


    final_prompt = f"""
    Refine the initial game prompt to improve decision-making based on the Game and Strategy.
    ##Initial prompt: {prompt}\n

    **Game:** {game_type}  
    **Strategy:**  
    - State Evaluation: {state_evaluation}  
    - Winning Strategy: {winning_strategy}  
    - Endgame Tactics: {endgame_tactics}  

    ### Instructions:
    1. The new prompt must **clearly guide decision-making**.
    2. It should **force the model to prioritize winning moves**.
    3. Language should be **direct, logical, and assertive**.
    4. Do NOT include the answer—only refine the prompt.
    5. Do NOT define the format of the output.

    ### Format Response as:

    {{
    "optimized_prompt": "string", // The refined prompt that clearly directs decision-making. }}
    """

    parsed_content = get_agent_response(agent, final_prompt, system_prompt="You are a game theorist and strategist.", temperature=spc_temperature)

    optimized_prompt = parsed_content.get("optimized_prompt")

    one_new_prompt = f"""{optimized_prompt}\n

    **Current State:**  
    - There are {remaining_items} items left.  
    - You can take 1 to 3 items per turn. 
    ### Instructions:
    1. **If a winning move exists, take it immediately.**  
    2. **Otherwise, follow optimal move principles.**  
    3. Justify your move using the extracted strategy.

    ### Format Response as:
    {{
    "reasoning": "string", // Explanation of the move based on the strategy.
    "action": integer // Chosen move (1, 2, or 3). }}
    """

    parsed_content = get_agent_response(agent, one_new_prompt, system_prompt="You are a skilled Nim player.")

    initial_reasoning = parsed_content.get("reasoning")
    action = parsed_content.get("action")

    initial_move = int(action)

    for k in range(num_refine):

        feedback_prompt = f"""
        #Game Role:\n You are {agent['name']}, a participant in a game of Nim variants.\n\n
        #Objective:\n Your goal is to win the game by taking all remaining items on your turn, leaving no items for your opponent. The person who takes the last item wins.\n\n
        #Game Rule:\n There is a single pile of items. You can take between 1 and {max_take} items on your turn.\n\n
        #Current State:\n There are {remaining_items} items remaining in the pile.\n\n
        #Task:\nBased on the current state of the game, give a feedback on the first trial's reasoning and action.\n\n
        #First trial's reasoning and action:\nYou initially chose {initial_move} items at first trial by the reason: '{initial_reasoning}'.\n\n

        The output should be a markdown code snippet formatted in the following schema, including the leading and trailing \\`\\`\\`json" and "\\`\\`\\`":\n\n```\n{{\n\t"feedback": string  // This is the feedback for the selected action and reasoning\n}}
        """

        parsed_content = get_agent_response(agent, feedback_prompt, system_prompt="You are a skilled Nim player.")
        feedback = parsed_content.get("feedback")

        refine_prompt = f"""
        #Game Role:\n You are {agent['name']}, a participant in a game of Nim variants.\n\n
        #Objective:\n Your goal is to win the game by taking all remaining items on your turn, leaving no items for your opponent. The person who takes the last item wins.\n\n
        #Game Rule:\n There is a single pile of items. You can take between 1 and {max_take} items on your turn.\n\n
        #Current State:\n There are {remaining_items} items remaining in the pile.\n\n

        You initially chose {initial_move} items at first trial by the reason: '{initial_reasoning}'.\n\n
        You recieved feedback on your action and reasoning: {feedback}\n\n
        #Task:\nBased on the current state of the game and the feedback, refine your reasoning and action. And finally, decide how many items you will take (between 1 and {max_take}) on this turn.\n\n

        The output should be a markdown code snippet formatted in the following schema, including the leading and trailing \\`\\`\\`json" and "\\`\\`\\`":\n\n```\n{{\n\t"reasoning": string  // This is the reasons for the action\n\t"action": integer  // This is an action you take based on the reasoning. Only provide integer between 1 and 3.\n}}
        """

        parsed_content = get_agent_response(agent, refine_prompt, system_prompt="You are a skilled Nim player.")
        refined_reasoning = parsed_content.get("reasoning")
        refined_action = parsed_content.get("action")

        if initial_move == int(refined_action):
            return refined_reasoning, refined_action
        else:
            initial_move = refined_action
            initial_reasoning = refined_reasoning

    return refined_reasoning, refined_action

def get_move_with_debate(agent1, agent2, remaining_items):
    initial_moves = {}
    initial_reasonings = {}
    i = 0
    for agent in [agent1, agent2]:
        prompt = f"""
        #Game Role:\n You are {agent['name']}, a participant in a game of Nim variants.\n\n
        #Objective:\n Your goal is to win the game by taking all remaining items on your turn, leaving no items for your opponent. The person who takes the last item wins.\n\n
        #Game Rule:\n There is a single pile of items. You can take between 1 and {max_take} items on your turn.\n\n
        #Current State:\n There are {remaining_items} items remaining in the pile.\n\n
        #Task:\nBased on the current state of the game, decide how many items you will take (between 1 and {max_take}) on this turn.\n\n

        The output should be a markdown code snippet formatted in the following schema, including the leading and trailing \\`\\`\\`json" and "\\`\\`\\`":\n\n```\n{{\n\t"reasoning": string  // This is the reasons for the action\n\t"action": integer  // This is an action you take based on the reasoning. Only provide integer between 1 and 3.\n}}
        """

        parsed_content = get_agent_response(agent, prompt, system_prompt="You are a skilled Nim player and debating the best move.")

        initial_reasoning = parsed_content.get("reasoning")
        initial_action = parsed_content.get("action")
        if i == 0:
            initial_moves['agent1'] = initial_action
            initial_reasonings['agent1'] = initial_reasoning
        if i == 1:
            initial_moves['agent2'] = initial_action
            initial_reasonings['agent2'] = initial_reasoning
        i += 1

    for _ in range(debate_rounds):
        i = 0
        for agent in [agent1, agent2]:
            
            if i == 0:
                prompt = f"""
                #Game Role:\n You are {agent['name']}, a participant in a game of Nim variants.\n\n
                #Objective:\n Your goal is to win the game by taking all remaining items on your turn, leaving no items for your opponent. The person who takes the last item wins.\n\n
                #Game Rule:\n There is a single pile of items. You can take between 1 and {max_take} items on your turn.\n\n
                #Current State:\n There are {remaining_items} items remaining in the pile.\n\n
                #Task:\nBased on the current state of the game and other agent's reasoning and action, decide how many items you will take (between 1 and {max_take}) on this turn.\n\n

                You initially chose {initial_moves['agent1']} items at first trial by the reason: '{initial_reasonings['agent1']}'.\n
                Other agent argues that you have to choose move as: {initial_moves['agent2']} by the reason: {initial_reasonings['agent2']}.\n
                Considering the other's opinion, refine or confirm your move.\n

                The output should be a markdown code snippet formatted in the following schema, including the leading and trailing \\`\\`\\`json" and "\\`\\`\\`":\n\n```\n{{\n\t"reasoning": string  // This is the reasons for the action\n\t"action": integer  // This is an action you take based on the reasoning. Only provide integer between 1 and 3.\n}}
                """
            if i == 1:
                prompt = f"""
                #Game Role:\n You are {agent['name']}, a participant in a game of Nim variants.\n\n
                #Objective:\n Your goal is to win the game by taking all remaining items on your turn, leaving no items for your opponent. The person who takes the last item wins.\n\n
                #Game Rule:\n There is a single pile of items. You can take between 1 and {max_take} items on your turn.\n\n
                #Current State:\n There are {remaining_items} items remaining in the pile.\n\n
                #Task:\nBased on the current state of the game and other agent's reasoning and action, decide how many items you will take (between 1 and {max_take}) on this turn.\n\n

                You initially chose {initial_moves['agent2']} items at first trial by the reason: '{initial_reasonings['agent2']}'.\n
                Other agent argues that you have to choose move as: {initial_moves['agent1']} by the reason: {initial_reasonings['agent1']}.\n
                Considering the other's opinion, refine or confirm your move.\n

                The output should be a markdown code snippet formatted in the following schema, including the leading and trailing \\`\\`\\`json" and "\\`\\`\\`":\n\n```\n{{\n\t"reasoning": string  // This is the reasons for the action\n\t"action": integer  // This is an action you take based on the reasoning. Only provide integer between 1 and 3.\n}}
                """

            parsed_content = get_agent_response(agent, prompt, system_prompt="You are a skilled Nim player and debating the best move.")

            initial_reasoning = parsed_content.get("reasoning")
            initial_action = parsed_content.get("action")
            if i == 0:
                a0_action = initial_action
                a0_reasoning = initial_reasoning
                
                # print('debate round:, ', t, 'my action: ', initial_action)
                # print('debate round:, ', t, 'my reasoning: ', initial_reasoning)    
            if i == 1:
                a1_action = initial_action
                a1_reasoning = initial_reasoning
                
                # print('debate round:, ', t, 'others action: ', initial_action)
                # print('debate round:, ', t, 'others reasoning: ', initial_reasoning)   
            i += 1
        initial_moves['agent1'] = a0_action
        initial_reasonings['agent1'] = a0_reasoning
        initial_moves['agent2'] = a1_action
        initial_reasonings['agent2'] = a1_reasoning
            
        if len(set(initial_moves.values())) == 1:
            return initial_reasoning, initial_action

    return initial_reasoning, Counter(initial_moves.values()).most_common(1)[0][0]  # Use most common if no consensus

def get_move_dreamad(agent1, agent2, remaining_items):
    initial_moves = {}
    initial_reasonings = {}
    i = 0
    for agent in [agent1]:
        prompt = f"""
        #Game Role:\n You are {agent['name']}, a participant in a game of Nim variants.\n\n
        #Objective:\n Your goal is to win the game by taking all remaining items on your turn, leaving no items for your opponent. The person who takes the last item wins.\n\n
        #Game Rule:\n There is a single pile of items. You can take between 1 and {max_take} items on your turn.\n\n
        #Current State:\n There are {remaining_items} items remaining in the pile.\n\n
        #Task:\nBased on the current state of the game, decide how many items you will take (between 1 and {max_take}) on this turn.\n
        """
        #################prompt1#################
        game_prompt = f"""
        Below is a game description. Extract key information.

        **Game Description:**
        {prompt}

        ### Format Response as:
        {{
        "game_type": "string", // Name of the game (if identifiable).
        "winning_condition": "string", // How to win the game.
        "move_constraints": "string" // What actions are allowed per turn.
        }}
        """
    
        parsed_content = get_agent_response(agent, game_prompt, system_prompt="You are a game theorist and strategist.",temperature=0.1)

        game_type = parsed_content.get("game_type")
        winning_condition = parsed_content.get("winning_condition")
        move_constraints = parsed_content.get("move_constraints")


        strategy_prompt = f"""
        Based on the game information below, derive the **optimal strategy**.

        **Game:** {game_type}  
        **Winning Condition:** {winning_condition}  
        **Move Constraints:** {move_constraints}

        ### Format Response as:
        {{
        "state_evaluation": "string", // How to assess the game state.
        "winning_strategy": "string", // Winning strategy in this turn to win this game.
        "endgame_tactics": "string" // Best strategy in a near-win situation.}}
        """
        # print("Strategy Prompt: ", strategy_prompt)
        parsed_content = get_agent_response(agent, strategy_prompt, system_prompt="You are a game theorist and strategist.", temperature=0.1)
        # print(2)

        state_evaluation = parsed_content.get("state_evaluation")
        winning_strategy = parsed_content.get("winning_strategy")
        endgame_tactics = parsed_content.get("endgame_tactics")


    for agent in [agent1, agent2]:

        final_prompt = f"""
        Refine the initial game prompt to improve decision-making based on the Game and Strategy.
        ##Initial prompt: {prompt}\n

        **Game:** {game_type}  
        **Strategy:**  
        - State Evaluation: {state_evaluation}  
        - Winning Strategy: {winning_strategy}  
        - Endgame Tactics: {endgame_tactics}  

        ### Instructions:
        1. The new prompt must **clearly guide decision-making**.
        2. It should **force the model to prioritize winning moves**.
        3. Language should be **direct, logical, and assertive**.
        4. Do NOT include the answer—only refine the prompt.
        5. Do NOT define the format of the output.

        ### Format Response as:

        {{
        "optimized_prompt": "string", // The refined prompt that clearly directs decision-making. }}
        """
        parsed_content = get_agent_response(agent, final_prompt, system_prompt="You are a game theorist and strategist.", temperature=spc_temperature)

        optimized_prompt = parsed_content.get("optimized_prompt")

        one_new_prompt = f"""{optimized_prompt}\n

        **Current State:**  
        - There are {remaining_items} items left.  
        - You can take 1 to 3 items per turn. 
        ### Instructions:
        1. **If a winning move exists, take it immediately.**  
        2. **Otherwise, follow optimal move principles.**  
        3. Justify your move using the extracted strategy.

        ### Format Response as:
        {{
        "reasoning": "string", // Explanation of the move based on the strategy.
        "action": integer // Chosen move (1, 2, or 3). }}
        """
        # print("One New Prompt: ", one_new_prompt)
        one_parsed_content = get_agent_response(agent, one_new_prompt, system_prompt="You are a game theorist and strategist.")
        # print(4)
        one_reasoning = one_parsed_content.get("reasoning")
        one_action = one_parsed_content.get("action")

        if i == 0:
            initial_moves['agent1'] = one_action
            initial_reasonings['agent1'] = one_reasoning
            one_prompt = optimized_prompt
        if i == 1:
            initial_moves['agent2'] = one_action
            initial_reasonings['agent2'] = one_reasoning
            two_prompt = optimized_prompt

        i += 1


    for _ in range(debate_rounds):
        i = 0
        for agent in [agent1, agent2]:
            
            if i == 0:
                prompt = f"""
                {one_prompt}\n

                You initially chose {initial_moves['agent1']} items at first trial by the reason: '{initial_reasonings['agent1']}'.\n
                Other agent argues that you have to choose move as: {initial_moves['agent2']} by the reason: {initial_reasonings['agent2']}.\n
                Considering the other's opinion and your strategy, refine or confirm your move.\n

                The output should be a markdown code snippet formatted in the following schema, including the leading and trailing \\`\\`\\`json" and "\\`\\`\\`":\n\n```\n{{\n\t"reasoning": string  // This is the reasons for the action\n\t"action": integer  // This is an action you take based on the reasoning. Only provide integer between 1 and 3.\n}}
                """
            if i == 1:
                prompt = f"""
                {two_prompt}\n

                You initially chose {initial_moves['agent2']} items at first trial by the reason: '{initial_reasonings['agent2']}'.\n
                Other agent argues that you have to choose move as: {initial_moves['agent1']} by the reason: {initial_reasonings['agent1']}.\n
                Considering the other's opinion and your strategy, refine or confirm your move.\n

                The output should be a markdown code snippet formatted in the following schema, including the leading and trailing \\`\\`\\`json" and "\\`\\`\\`":\n\n```\n{{\n\t"reasoning": string  // This is the reasons for the action\n\t"action": integer  // This is an action you take based on the reasoning. Only provide integer between 1 and 3.\n}}
                """

            parsed_content = get_agent_response(agent, prompt, system_prompt="You are a skilled Game player and debating the best move.")

            initial_reasoning = parsed_content.get("reasoning")
            initial_action = parsed_content.get("action")
            if i == 0:
                a0_action = initial_action
                a0_reasoning = initial_reasoning
                
                # print('debate round:, ', t, 'my action: ', initial_action)
                # print('debate round:, ', t, 'my reasoning: ', initial_reasoning)    
            if i == 1:
                a1_action = initial_action
                a1_reasoning = initial_reasoning
                
                # print('debate round:, ', t, 'others action: ', initial_action)
                # print('debate round:, ', t, 'others reasoning: ', initial_reasoning)   
            i += 1
        initial_moves['agent1'] = a0_action
        initial_reasonings['agent1'] = a0_reasoning
        initial_moves['agent2'] = a1_action
        initial_reasonings['agent2'] = a1_reasoning
            
        if len(set(initial_moves.values())) == 1:
            return initial_reasoning, initial_action

    return initial_reasoning, Counter(initial_moves.values()).most_common(1)[0][0]  # Use most common if no consensus

def get_move_dreamad_three(agent1, agent2, agent3, remaining_items):
    initial_moves = {}
    initial_reasonings = {}
    i = 0
    for agent in [agent1]:
        prompt = f"""
        #Game Role:\n You are {agent['name']}, a participant in a game of Nim variants.\n\n
        #Objective:\n Your goal is to win the game by taking all remaining items on your turn, leaving no items for your opponent. The person who takes the last item wins.\n\n
        #Game Rule:\n There is a single pile of items. You can take between 1 and {max_take} items on your turn.\n\n
        #Current State:\n There are {remaining_items} items remaining in the pile.\n\n
        #Task:\nBased on the current state of the game, decide how many items you will take (between 1 and {max_take}) on this turn.\n
        """
        #################prompt1#################
        game_prompt = f"""
        Below is a game description. Extract key information.

        **Game Description:**
        {prompt}

        ### Format Response as:
        {{
        "game_type": "string", // Name of the game (if identifiable).
        "winning_condition": "string", // How to win the game.
        "move_constraints": "string" // What actions are allowed per turn.
        }}
        """
    
        parsed_content = get_agent_response(agent, game_prompt, system_prompt="You are a game theorist and strategist.",temperature=0.1)

        game_type = parsed_content.get("game_type")
        winning_condition = parsed_content.get("winning_condition")
        move_constraints = parsed_content.get("move_constraints")


        strategy_prompt = f"""
        Based on the game information below, derive the **optimal strategy**.

        **Game:** {game_type}  
        **Winning Condition:** {winning_condition}  
        **Move Constraints:** {move_constraints}

        ### Format Response as:
        {{
        "state_evaluation": "string", // How to assess the game state.
        "winning_strategy": "string", // Winning strategy in this turn to win this game.
        "endgame_tactics": "string" // Best strategy in a near-win situation.}}
        """
        # print("Strategy Prompt: ", strategy_prompt)
        parsed_content = get_agent_response(agent, strategy_prompt, system_prompt="You are a game theorist and strategist.", temperature=0.1)
        # print(2)

        state_evaluation = parsed_content.get("state_evaluation")
        winning_strategy = parsed_content.get("winning_strategy")
        endgame_tactics = parsed_content.get("endgame_tactics")


    for agent in [agent1, agent2, agent3]:

        final_prompt = f"""
        Refine the initial game prompt to improve decision-making based on the Game and Strategy.
        ##Initial prompt: {prompt}\n

        **Game:** {game_type}  
        **Strategy:**  
        - State Evaluation: {state_evaluation}  
        - Winning Strategy: {winning_strategy}  
        - Endgame Tactics: {endgame_tactics}  

        ### Instructions:
        1. The new prompt must **clearly guide decision-making**.
        2. It should **force the model to prioritize winning moves**.
        3. Language should be **direct, logical, and assertive**.
        4. Do NOT include the answer—only refine the prompt.
        5. Do NOT define the format of the output.

        ### Format Response as:

        {{
        "optimized_prompt": "string", // The refined prompt that clearly directs decision-making. }}
        """
        parsed_content = get_agent_response(agent, final_prompt, system_prompt="You are a game theorist and strategist.", temperature=spc_temperature)

        optimized_prompt = parsed_content.get("optimized_prompt")

        one_new_prompt = f"""{optimized_prompt}\n

        **Current State:**  
        - There are {remaining_items} items left.  
        - You can take 1 to 3 items per turn. 
        ### Instructions:
        1. **If a winning move exists, take it immediately.**  
        2. **Otherwise, follow optimal move principles.**  
        3. Justify your move using the extracted strategy.

        ### Format Response as:
        {{
        "reasoning": "string", // Explanation of the move based on the strategy.
        "action": integer // Chosen move (1, 2, or 3). }}
        """
        # print("One New Prompt: ", one_new_prompt)
        one_parsed_content = get_agent_response(agent, one_new_prompt, system_prompt="You are a game theorist and strategist.")
        # print(4)
        one_reasoning = one_parsed_content.get("reasoning")
        one_action = one_parsed_content.get("action")

        if i == 0:
            initial_moves['agent1'] = one_action
            initial_reasonings['agent1'] = one_reasoning
            one_prompt = optimized_prompt
        if i == 1:
            initial_moves['agent2'] = one_action
            initial_reasonings['agent2'] = one_reasoning
            two_prompt = optimized_prompt
        if i == 2:
            initial_moves['agent3'] = one_action
            initial_reasonings['agent3'] = one_reasoning
            three_prompt = optimized_prompt


        i += 1


    for _ in range(debate_rounds):
        i = 0
        for agent in [agent1, agent2, agent3]:
            
            if i == 0:
                prompt = f"""
                {one_prompt}\n

                You initially chose {initial_moves['agent1']} items at first trial by the reason: '{initial_reasonings['agent1']}'.\n
                One agent argues that you have to choose move as: {initial_moves['agent2']} by the reason: {initial_reasonings['agent2']}.\n
                Another agent argues that you have to choose move as: {initial_moves['agent3']} by the reason: {initial_reasonings['agent3']}.\n
                Considering the other's opinion and your strategy, refine or confirm your move.\n

                The output should be a markdown code snippet formatted in the following schema, including the leading and trailing \\`\\`\\`json" and "\\`\\`\\`":\n\n```\n{{\n\t"reasoning": string  // This is the reasons for the action\n\t"action": integer  // This is an action you take based on the reasoning. Only provide integer between 1 and 3.\n}}
                """
            if i == 1:
                prompt = f"""
                {two_prompt}\n

                You initially chose {initial_moves['agent2']} items at first trial by the reason: '{initial_reasonings['agent2']}'.\n
                One agent argues that you have to choose move as: {initial_moves['agent1']} by the reason: {initial_reasonings['agent1']}.\n
                Another agent argues that you have to choose move as: {initial_moves['agent3']} by the reason: {initial_reasonings['agent3']}.\n
                Considering the other's opinion and your strategy, refine or confirm your move.\n
                

                The output should be a markdown code snippet formatted in the following schema, including the leading and trailing \\`\\`\\`json" and "\\`\\`\\`":\n\n```\n{{\n\t"reasoning": string  // This is the reasons for the action\n\t"action": integer  // This is an action you take based on the reasoning. Only provide integer between 1 and 3.\n}}
                """
            
            if i == 2:
                prompt = f"""
                {three_prompt}\n

                You initially chose {initial_moves['agent3']} items at first trial by the reason: '{initial_reasonings['agent3']}'.\n
                One agent argues that you have to choose move as: {initial_moves['agent1']} by the reason: {initial_reasonings['agent1']}.\n
                Another agent argues that you have to choose move as: {initial_moves['agent2']} by the reason: {initial_reasonings['agent2']}.\n
                Considering the other's opinion and your strategy, refine or confirm your move.\n

                The output should be a markdown code snippet formatted in the following schema, including the leading and trailing \\`\\`\\`json" and "\\`\\`\\`":\n\n```\n{{\n\t"reasoning": string  // This is the reasons for the action\n\t"action": integer  // This is an action you take based on the reasoning. Only provide integer between 1 and 3.\n}}
                """

            parsed_content = get_agent_response(agent, prompt, system_prompt="You are a skilled Game player and debating the best move.")

            initial_reasoning = parsed_content.get("reasoning")
            initial_action = parsed_content.get("action")
            if i == 0:
                a0_action = initial_action
                a0_reasoning = initial_reasoning
                
            if i == 1:
                a1_action = initial_action
                a1_reasoning = initial_reasoning
            
            if i == 2:
                a2_action = initial_action
                a2_reasoning = initial_reasoning
                
            i += 1
        initial_moves['agent1'] = a0_action
        initial_reasonings['agent1'] = a0_reasoning
        initial_moves['agent2'] = a1_action
        initial_reasonings['agent2'] = a1_reasoning
        initial_moves['agent3'] = a2_action
        initial_reasonings['agent3'] = a2_reasoning
            
        if len(set(initial_moves.values())) == 1:
            return initial_reasoning, initial_action

    return initial_reasoning, Counter(initial_moves.values()).most_common(1)[0][0]  # Use most common if no consensus

def get_move_dreamad_one(agent1, agent2, remaining_items):
    initial_moves = {}
    initial_reasonings = {}
    i = 0
    for agent in [agent1]:
        prompt = f"""
        #Game Role:\n You are {agent['name']}, a participant in a game of Nim variants.\n\n
        #Objective:\n Your goal is to win the game by taking all remaining items on your turn, leaving no items for your opponent. The person who takes the last item wins.\n\n
        #Game Rule:\n There is a single pile of items. You can take between 1 and {max_take} items on your turn.\n\n
        #Current State:\n There are {remaining_items} items remaining in the pile.\n\n
        #Task:\nBased on the current state of the game, decide how many items you will take (between 1 and {max_take}) on this turn.\n
        """
        #################prompt1#################
        game_prompt = f"""
        Below is a game description. Extract key information.

        **Game Description:**
        {prompt}

        ### Format Response as:
        {{
        "game_type": "string", // Name of the game (if identifiable).
        "winning_condition": "string", // How to win the game.
        "move_constraints": "string" // What actions are allowed per turn.
        }}
        """
    
        parsed_content = get_agent_response(agent, game_prompt, system_prompt="You are a game theorist and strategist.",temperature=0.1)

        game_type = parsed_content.get("game_type")
        winning_condition = parsed_content.get("winning_condition")
        move_constraints = parsed_content.get("move_constraints")


        strategy_prompt = f"""
        Based on the game information below, derive the **optimal strategy**.

        **Game:** {game_type}  
        **Winning Condition:** {winning_condition}  
        **Move Constraints:** {move_constraints}

        ### Format Response as:
        {{
        "state_evaluation": "string", // How to assess the game state.
        "winning_strategy": "string", // Winning strategy in this turn to win this game.
        "endgame_tactics": "string" // Best strategy in a near-win situation.}}
        """
        # print("Strategy Prompt: ", strategy_prompt)
        parsed_content = get_agent_response(agent, strategy_prompt, system_prompt="You are a game theorist and strategist.", temperature=0.1)
        # print(2)

        state_evaluation = parsed_content.get("state_evaluation")
        winning_strategy = parsed_content.get("winning_strategy")
        endgame_tactics = parsed_content.get("endgame_tactics")

        final_prompt = f"""
        Refine the initial game prompt to improve decision-making based on the Game and Strategy.
        ##Initial prompt: {prompt}\n

        **Game:** {game_type}  
        **Strategy:**  
        - State Evaluation: {state_evaluation}  
        - Winning Strategy: {winning_strategy}  
        - Endgame Tactics: {endgame_tactics}  

        ### Instructions:
        1. The new prompt must **clearly guide decision-making**.
        2. It should **force the model to prioritize winning moves**.
        3. Language should be **direct, logical, and assertive**.
        4. Do NOT include the answer—only refine the prompt.
        5. Do NOT define the format of the output.

        ### Format Response as:

        {{
        "optimized_prompt": "string", // The refined prompt that clearly directs decision-making. }}
        """
        parsed_content = get_agent_response(agent, final_prompt, system_prompt="You are a game theorist and strategist.", temperature=spc_temperature)

        optimized_prompt = parsed_content.get("optimized_prompt")

        one_new_prompt = f"""{optimized_prompt}\n

        **Current State:**  
        - There are {remaining_items} items left.  
        - You can take 1 to 3 items per turn. 
        ### Instructions:
        1. **If a winning move exists, take it immediately.**  
        2. **Otherwise, follow optimal move principles.**  
        3. Justify your move using the extracted strategy.

        ### Format Response as:
        {{
        "reasoning": "string", // Explanation of the move based on the strategy.
        "action": integer // Chosen move (1, 2, or 3). }}
        """


    for agent in [agent1, agent2]:

        one_parsed_content = get_agent_response(agent, one_new_prompt, system_prompt="You are a game theorist and strategist.")
        # print(4)
        one_reasoning = one_parsed_content.get("reasoning")
        one_action = one_parsed_content.get("action")

        if i == 0:
            initial_moves['agent1'] = one_action
            initial_reasonings['agent1'] = one_reasoning
        if i == 1:
            initial_moves['agent2'] = one_action
            initial_reasonings['agent2'] = one_reasoning

        i += 1


    for _ in range(debate_rounds):
        i = 0
        for agent in [agent1, agent2]:
            
            if i == 0:
                prompt = f"""
                {optimized_prompt}\n

                You initially chose {initial_moves['agent1']} items at first trial by the reason: '{initial_reasonings['agent1']}'.\n
                Other agent argues that you have to choose move as: {initial_moves['agent2']} by the reason: {initial_reasonings['agent2']}.\n
                Considering the other's opinion and your strategy, refine or confirm your move.\n

                The output should be a markdown code snippet formatted in the following schema, including the leading and trailing \\`\\`\\`json" and "\\`\\`\\`":\n\n```\n{{\n\t"reasoning": string  // This is the reasons for the action\n\t"action": integer  // This is an action you take based on the reasoning. Only provide integer between 1 and 3.\n}}
                """
            if i == 1:
                prompt = f"""
                {optimized_prompt}\n

                You initially chose {initial_moves['agent2']} items at first trial by the reason: '{initial_reasonings['agent2']}'.\n
                Other agent argues that you have to choose move as: {initial_moves['agent1']} by the reason: {initial_reasonings['agent1']}.\n
                Considering the other's opinion and your strategy, refine or confirm your move.\n

                The output should be a markdown code snippet formatted in the following schema, including the leading and trailing \\`\\`\\`json" and "\\`\\`\\`":\n\n```\n{{\n\t"reasoning": string  // This is the reasons for the action\n\t"action": integer  // This is an action you take based on the reasoning. Only provide integer between 1 and 3.\n}}
                """

            parsed_content = get_agent_response(agent, prompt, system_prompt="You are a skilled Game player and debating the best move.")

            initial_reasoning = parsed_content.get("reasoning")
            initial_action = parsed_content.get("action")
            if i == 0:
                a0_action = initial_action
                a0_reasoning = initial_reasoning
                
                # print('debate round:, ', t, 'my action: ', initial_action)
                # print('debate round:, ', t, 'my reasoning: ', initial_reasoning)    
            if i == 1:
                a1_action = initial_action
                a1_reasoning = initial_reasoning
                
                # print('debate round:, ', t, 'others action: ', initial_action)
                # print('debate round:, ', t, 'others reasoning: ', initial_reasoning)   
            i += 1
        initial_moves['agent1'] = a0_action
        initial_reasonings['agent1'] = a0_reasoning
        initial_moves['agent2'] = a1_action
        initial_reasonings['agent2'] = a1_reasoning
            
        if len(set(initial_moves.values())) == 1:
            return initial_reasoning, initial_action

    return initial_reasoning, Counter(initial_moves.values()).most_common(1)[0][0]  # Use most common if no consensus

def get_move_with_bias_mitigate_debate(agent1, agent2, remaining_items):
    initial_moves = {}
    initial_reasonings = {}
    i = 0
    for agent in [agent1, agent2]:
        if i == 0:
            prompt = f"""
            #Game Role:\n You are {agent['name']}, a participant in a game of Nim variants.\n\n
            #Objective:\n Your goal is to win the game by taking all remaining items on your turn, leaving no items for your opponent. The person who takes the last item wins.\n\n
            #Game Rule:\n There is a single pile of items. You can take between 1 and {max_take} items on your turn.\n\n
            #Current State:\n There are {remaining_items} items remaining in the pile.\n\n
            #Task:\nBased on the current state of the game, decide how many items you will take (between 1 and {max_take}) on this turn.\n\n

            The output should be a markdown code snippet formatted in the following schema, including the leading and trailing \\`\\`\\`json" and "\\`\\`\\`":\n\n```\n{{\n\t"reasoning": string  // This is the reasons for the action\n\t"action": integer  // This is an action you take based on the reasoning. Only provide integer between 1 and 3.\n}}
            """

            parsed_content = get_agent_response(agent, prompt, system_prompt="You are a skilled Nim player and debating the best move.")

            initial_reasoning = parsed_content.get("reasoning")
            initial_action = parsed_content.get("action")
        if i == 1:
            
            text = f"""Given the following instruction, rewrite it to minimize bias stemming from strong prior knowledge while preserving its original intent and clarity.\n
            #Instruction:{prompt}\n
                

            The output should be a markdown code snippet formatted in the following schema, including the leading and trailing \\`\\`\\`json" and "\\`\\`\\`":\n\n```\n{{\n\t"instruction": string  // This is a rewritten instruction to minimize the bias.\n}}
            """

            parsed_content = get_agent_response(agent, text, system_prompt="You are a rational smart assistant.")
            new_instruction = parsed_content.get("instruction")

            text = f"""{new_instruction}\n\n

            The output should be a markdown code snippet formatted in the following schema, including the leading and trailing \\`\\`\\`json" and "\\`\\`\\`":\n\n```\n{{\n\t"reasoning": string  // This is the reasons for the action\n\t"action": integer  // This is an action you take based on the reasoning. Only provide integer between 1 and 3.}}
            """

            parsed_content = get_agent_response(agent, text, system_prompt="You are a rational game player.")

            initial_reasoning = parsed_content.get("reasoning")
            initial_action = parsed_content.get("action")

        if i == 0:
            initial_moves['agent1'] = initial_action
            initial_reasonings['agent1'] = initial_reasoning
        if i == 1:
            initial_moves['agent2'] = initial_action
            initial_reasonings['agent2'] = initial_reasoning
        i += 1

    for _ in range(debate_rounds):
        i = 0
        for agent in [agent1, agent2]:
            if i == 0:
                prompt = f"""
                #Game Role:\n You are {agent['name']}, a participant in a game of Nim variants.\n\n
                #Objective:\n Your goal is to win the game by taking all remaining items on your turn, leaving no items for your opponent. The person who takes the last item wins.\n\n
                #Game Rule:\n There is a single pile of items. You can take between 1 and {max_take} items on your turn.\n\n
                #Current State:\n There are {remaining_items} items remaining in the pile.\n\n
                #Task:\nBased on the current state of the game and other agent's reasoning and action, decide how many items you will take (between 1 and {max_take}) on this turn.\n\n

                You initially chose {initial_moves['agent1']} items at first trial by the reason: '{initial_reasonings['agent1']}'.\n
                Other agent argues that you have to choose move as: {initial_moves['agent2']} by the reason: {initial_reasonings['agent2']}.\n
                Considering the other's opinion, refine or confirm your move.\n

                The output should be a markdown code snippet formatted in the following schema, including the leading and trailing \\`\\`\\`json" and "\\`\\`\\`":\n\n```\n{{\n\t"reasoning": string  // This is the reasons for the action\n\t"action": integer  // This is an action you take based on the reasoning. Only provide integer between 1 and 3.\n}}
                """
            if i == 1:
                prompt = f"""
                {new_instruction}\n\n

                You initially chose {initial_moves['agent2']} items at first trial by the reason: '{initial_reasonings['agent2']}'.\n
                Other agent argues that you have to choose move as: {initial_moves['agent1']} by the reason: {initial_reasonings['agent1']}.\n
                Considering the other's opinion, refine or confirm your move.\n

                The output should be a markdown code snippet formatted in the following schema, including the leading and trailing \\`\\`\\`json" and "\\`\\`\\`":\n\n```\n{{\n\t"reasoning": string  // This is the reasons for the action\n\t"action": integer  // This is an action you take based on the reasoning. Only provide integer between 1 and 3.\n}}
                """

            parsed_content = get_agent_response(agent, prompt, system_prompt="You are a skilled Nim player and debating the best move.")

            initial_reasoning = parsed_content.get("reasoning")
            initial_action = parsed_content.get("action")
            if i == 0:
                initial_moves['agent1'] = initial_action
                initial_reasonings['agent1'] = initial_reasoning
            if i == 1:
                initial_moves['agent2'] = initial_action
                initial_reasonings['agent2'] = initial_reasoning

            i += 1
    
        if len(set(initial_moves.values())) == 1:
            return initial_reasoning, initial_action

    return initial_reasoning, Counter(initial_moves.values()).most_common(1)[0][0]  # Use most common if no consensus

def get_move_with_original_vs_harmonized_debate(agent1, agent2, remaining_items):
    initial_moves = {}
    initial_reasonings = {}
    i = 0
    for agent in [agent1, agent2]:
        if i == 0:
            prompt = f"""
            #Game Role:\n You are {agent['name']}, a participant in a game of Nim variants.\n\n
            #Objective:\n Your goal is to win the game by taking all remaining items on your turn, leaving no items for your opponent. The person who takes the last item wins.\n\n
            #Game Rule:\n There is a single pile of items. You can take between 1 and {max_take} items on your turn.\n\n
            #Current State:\n There are {remaining_items} items remaining in the pile.\n\n
            #Task:\nBased on the current state of the game, decide how many items you will take (between 1 and {max_take}) on this turn.\n\n

            The output should be a markdown code snippet formatted in the following schema, including the leading and trailing \\`\\`\\`json" and "\\`\\`\\`":\n\n```\n{{\n\t"reasoning": string  // This is the reasons for the action\n\t"action": integer  // This is an action you take based on the reasoning. Only provide integer between 1 and 3.\n}}
            """

            parsed_content = get_agent_response(agent, prompt, system_prompt="You are a skilled Nim player and debating the best move.")

            initial_reasoning = parsed_content.get("reasoning")
            initial_action = parsed_content.get("action")
        if i == 1:
            
            text = f"""Given the following instruction, rewrite it to minimize bias stemming from strong prior knowledge while preserving its original intent and clarity.\n
            #Instruction:{prompt}\n
                

            The output should be a markdown code snippet formatted in the following schema, including the leading and trailing \\`\\`\\`json" and "\\`\\`\\`":\n\n```\n{{\n\t"instruction": string  // This is a rewritten instruction to minimize the bias.\n \t"reasoning": string  // This is the reason why new instruction minimize bias from the original prompt.}}
            """

            parsed_content = get_agent_response(agent, text, system_prompt="You are a rational smart assistant.")
            new_instruction = parsed_content.get("instruction")

            text = f"""Combine the following two instructions into a single instruction that captures their shared intention while harmonizing their nuances. Pay attention to clarity and ensure that any biases in the original instructions are mitigated.

        - Original instruction:\n (`{prompt}`): The first instruction to consider.
        - Bias-mitigated instruction:\n (`{new_instruction}`): The second instruction to harmonize.

            The output should be a markdown code snippet formatted in the following schema, including the leading and trailing \\`\\`\\`json" and "\\`\\`\\`":\n\n```\n{{\n\t"instruction": string  // This is the combined instruction harmonizing the two instructions.\n\t"reasoning": string  // This is the reason why new instruction is harmonized.}}
            """

            parsed_content = get_agent_response(agent, text, system_prompt="You are a rational smart assistant.")

            new_instruction = parsed_content.get("instruction")

            text = f"""{new_instruction}\n\n

            The output should be a markdown code snippet formatted in the following schema, including the leading and trailing \\`\\`\\`json" and "\\`\\`\\`":\n\n```\n{{\n\t"reasoning": string  // This is the reasons for the action\n\t"action": integer  // This is an action you take based on the reasoning. Only provide integer between 1 and 3. You cannot choose 0.}}
            """

            parsed_content = get_agent_response(agent, text, system_prompt="You are a rational game player.")

            initial_reasoning = parsed_content.get("reasoning")
            initial_action = parsed_content.get("action")

        if i == 0:
            initial_moves['agent1'] = initial_action
            initial_reasonings['agent1'] = initial_reasoning
        if i == 1:
            initial_moves['agent2'] = initial_action
            initial_reasonings['agent2'] = initial_reasoning
        i += 1

    for _ in range(debate_rounds):
        i = 0
        for agent in [agent1, agent2]:
            if i == 0:
                prompt = f"""
                #Game Role:\n You are {agent['name']}, a participant in a game of Nim variants.\n\n
                #Objective:\n Your goal is to win the game by taking all remaining items on your turn, leaving no items for your opponent. The person who takes the last item wins.\n\n
                #Game Rule:\n There is a single pile of items. You can take between 1 and {max_take} items on your turn.\n\n
                #Current State:\n There are {remaining_items} items remaining in the pile.\n\n
                #Task:\nBased on the current state of the game and other agent's reasoning and action, decide how many items you will take (between 1 and {max_take}) on this turn.\n\n

                You initially chose {initial_moves['agent1']} items at first trial by the reason: '{initial_reasonings['agent1']}'.\n
                Other agent argues that you have to choose move as: {initial_moves['agent2']} by the reason: {initial_reasonings['agent2']}.\n
                Considering the other's opinion, refine or confirm your move.\n

                The output should be a markdown code snippet formatted in the following schema, including the leading and trailing \\`\\`\\`json" and "\\`\\`\\`":\n\n```\n{{\n\t"reasoning": string  // This is the reasons for the action\n\t"action": integer  // This is an action you take based on the reasoning. Only provide integer between 1 and 3.\n}}
                """
            if i == 1:
                prompt = f"""
                {new_instruction}\n\n

                You initially chose {initial_moves['agent2']} items at first trial by the reason: '{initial_reasonings['agent2']}'.\n
                Other agent argues that you have to choose move as: {initial_moves['agent1']} by the reason: {initial_reasonings['agent1']}.\n
                Considering the other's opinion, refine or confirm your move.\n

                The output should be a markdown code snippet formatted in the following schema, including the leading and trailing \\`\\`\\`json" and "\\`\\`\\`":\n\n```\n{{\n\t"reasoning": string  // This is the reasons for the action\n\t"action": integer  // This is an action you take based on the reasoning. Only provide integer between 1 and 3.\n}}
                """

            parsed_content = get_agent_response(agent, prompt, system_prompt="You are a skilled game player and debating the best move.")

            initial_reasoning = parsed_content.get("reasoning")
            initial_action = parsed_content.get("action")
            if i == 0:
                initial_moves['agent1'] = initial_action
                initial_reasonings['agent1'] = initial_reasoning
            if i == 1:
                initial_moves['agent2'] = initial_action
                initial_reasonings['agent2'] = initial_reasoning

            i += 1
    
        if len(set(initial_moves.values())) == 1:
            return initial_reasoning, initial_action

    return initial_reasoning, Counter(initial_moves.values()).most_common(1)[0][0]  # Use most common if no consensus



def play_nim_game(total_items, max_take, verbose=False):
    with open(f'/home/jihwan/NashIP/result/BR31/{args.agent1_model}_{args.agent1_prompt}_{n_step_lookahead}_{spc_temperature}_{args.agent2_model}_{args.agent2_prompt}.txt', 'a') as f:
        current_items = total_items
        turn = 0
        while current_items > 0:
            current_agent = agents[turn % 2]
            other_agent = agents[(turn + 1) % 2]

            if current_agent["prompting_method"] == "self_consistency":
                reasoning, move = get_consistent_move(current_agent, current_items, self_consistency_count)
            elif current_agent["prompting_method"] == "diverse_consistency":
                reasoning, move = get_consistent_diverse_move(current_agent, current_items, self_consistency_count)
            elif current_agent["prompting_method"] == "simple":
                reasoning, move = get_move(current_agent, current_items) 
            elif current_agent["prompting_method"] == "self_reflection":
                reasoning, move = get_move_with_reflection(current_agent, current_items)
            elif current_agent["prompting_method"] == "diverse_reflection":
                reasoning, move = get_move_with_diverse_reflection(current_agent, current_items)
            elif current_agent["prompting_method"] == "debate":
                reasoning, move = get_move_with_debate(current_agent, current_agent, current_items)
            elif current_agent["prompting_method"] == "dreamad":
                reasoning, move = get_move_dreamad(current_agent, current_agent, current_items)
            elif current_agent["prompting_method"] == "dreamad_three":
                reasoning, move = get_move_dreamad_three(current_agent, current_agent, current_agent, current_items)
            elif current_agent["prompting_method"] == "dreamad_one":
                reasoning, move = get_move_dreamad_one(current_agent, current_agent, current_items)
            elif current_agent["prompting_method"] == "basic":
                reasoning, move = get_basic_move(current_agent, current_items)
            elif current_agent["prompting_method"] == "bias_mitigate_debate":
                reasoning, move = get_move_with_bias_mitigate_debate(current_agent, current_agent, current_items)
            elif current_agent["prompting_method"] == "original_vs_harmonized_debate":
                reasoning, move = get_move_with_original_vs_harmonized_debate(current_agent, current_agent, current_items)
            else:
                print("Error: set the prompting methods")

            if verbose:
                print('Reasoning:', reasoning, '\nAction:', move, file = f)
                print(f"{current_agent['name']} ({current_agent['model']}) takes {move} items. Items remaining: {current_items - move}", file = f)
            current_items -= move

            if current_items <= 0:
                if verbose:
                    print(f"{current_agent['name']} ({current_agent['model']}) wins!", file = f)
                return current_agent["name"]

            turn += 1

# Run the simulation
def simulate_games(num_games, total_items, max_take):
    win_counts = {agent["name"]: 0 for agent in agents}
    with open(f'/home/jihwan/NashIP/result/BR31/{args.agent1_model}_{args.agent1_prompt}_{n_step_lookahead}_{spc_temperature}_{args.agent2_model}_{args.agent2_prompt}.txt', 'a') as f:
        for game_num in range(num_games):
            print(f"\nStarting Game {game_num + 1}", file = f)
            print(f"\nStarting Game {game_num + 1}")
            winner = play_nim_game(total_items, max_take, verbose=True)
            win_counts[winner] += 1
        
        print("\nGame Results:", file = f)
        for agent in agents:
            win_rate = (win_counts[agent["name"]] / num_games) * 100
            print(f"{agent['name']} Win Rate: {win_rate:.2f}% ({win_counts[agent['name']]} wins out of {num_games})", file = f)

simulate_games(num_games, total_items, max_take)