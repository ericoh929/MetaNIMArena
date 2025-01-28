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
num_games = args.num_games  # Number of games to play
num_refine = 3
self_consistency_count = 10  # Number of responses to use for self-consistency
n_step_lookahead = args.look_ahead  # Number of lookahead steps for n-step opponent modeling
debate_rounds = 3  # Maximum number of debate rounds
max_retries = 5

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

def get_agent_response(agent, prompt, system_prompt="You are a skilled Nim player."):
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
            if agent["model"] == "llama":
                split_key = "<|start_header_id|>assistant<|end_header_id|>\n"
                response = say_model(system_prompt, prompt)
                parsed_content = parse_content(response, split_key)
                if parsed_content:
                    return parsed_content

            elif agent["model"] == "gemma":
                split_key = "<start_of_turn>model\n"
                response = say_model(system_prompt, prompt)
                parsed_content = parse_content(response, split_key)
                if parsed_content:
                    return parsed_content

            elif agent["model"] in ["gemini-1.5-flash", "gemini-1.5-pro"]:
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
                    system_instruction=system_prompt,
                )
                chat_session = model.start_chat(history=[])
                response = chat_session.send_message(prompt)
                try:
                    return json.loads(response.text)
                except json.JSONDecodeError:
                    print("Error decoding JSON for gemini model. Retrying...")

            elif agent["model"] in ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"]:
                response = client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    model=agent["model"],
                    temperature=0.7,
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


agents = [
    {"name": "Agent 1", "model": args.agent1_model, "prompting_method": args.agent1_prompt},
    {"name": "Agent 2", "model": args.agent2_model, "prompting_method": args.agent2_prompt}
]

# Function for basic move (single-response without consistency or modeling)
def get_basic_move(agent, grid):
    remaining_grid = [
        ''.join(['1' if cell else '0' for cell in row]) for row in grid
    ]
    prompt = f"""
    # Game Role:
    You are {agent['name']}, a participant in a game of Chomp.

    # Objective:
    Your goal is to force your opponent to take the top-left corner of the grid (position (0, 0)).

    # Game Rule:
    1. The game is played on a square grid.
    2. On your turn, you select a position (row, col).
    3. All positions to the right and below the selected position are removed.
    4. The player forced to select (0, 0) loses.

    # Current State:
    The grid is represented as binary strings, where '1' means the position is still available, and '0' means it is removed:
    {remaining_grid}

    # Task:
    Based on the current state of the grid, decide which position (row, col) you will select.

    # Output:
    Provide your reasoning for the move and the action in the following JSON format:

    ```
    {{
        "reasoning": string,  // Explain why you chose this position.
        "row": integer,       // The row index of the position you select (0-based).
        "col": integer        // The column index of the position you select (0-based).
    }}
    ```
    """

    retries = 0
    while retries < max_retries:
        parsed_content = get_agent_response(agent, prompt, system_prompt="You are a skilled Chomp player.")
        reasoning = parsed_content.get("reasoning")
        row = parsed_content.get("row")
        col = parsed_content.get("col")

        if is_valid_move_chomp(grid, row, col):
            return reasoning, (row, col)

        retries += 1

    # Fallback: Randomly select a valid position
    available_positions = [
        (r, c) for r, row in enumerate(grid) for c, cell in enumerate(row) if cell
    ]
    if available_positions:
        row, col = random.choice(available_positions)
        reasoning = "Fallback: Randomly selected a position after multiple failed attempts."
        return reasoning, (row, col)

    raise ValueError("No valid moves available, and fallback failed.")


# Function for self-consistency: generate multiple responses and choose the most common move
def get_consistent_move(agent, grid, num_responses):
    remaining_grid = [
        ''.join(['1' if cell else '0' for cell in row]) for row in grid
    ]
    prompt = f"""
    # Game Role:
    You are {agent['name']}, a participant in a game of Chomp.

    # Objective:
    Your goal is to force your opponent to take the top-left corner of the grid (position (0, 0)).

    # Game Rule:
    1. The game is played on a square grid.
    2. On your turn, you select a position (row, col).
    3. All positions to the right and below the selected position are removed.
    4. The player forced to select (0, 0) loses.

    # Current State:
    The grid is represented as binary strings, where '1' means the position is still available, and '0' means it is removed:
    {remaining_grid}

    # Task:
    Based on the current state of the grid, decide which position (row, col) you will select.

    # Output:
    Provide your reasoning for the move and the action in the following JSON format:

    ```
    {{
        "reasoning": string,  // Explain why you chose this position.
        "row": integer,       // The row index of the position you select (0-based).
        "col": integer        // The column index of the position you select (0-based).
    }}
    ```
    """

    moves = []
    reasoning_list = []

    for _ in range(num_responses):

        retries = 0
        while retries < max_retries:
            parsed_content = get_agent_response(agent, prompt, system_prompt="You are a skilled Chomp player.")
            reasoning = parsed_content.get("reasoning")
            row = parsed_content.get("row")
            col = parsed_content.get("col")

            if is_valid_move_chomp(grid, row, col):
                break

            retries += 1

        if retries == max_retries:
            available_positions = [
                (r, c) for r, row_data in enumerate(grid) for c, cell in enumerate(row_data) if cell
            ]
            if available_positions:
                row, col = random.choice(available_positions)
                reasoning = "Fallback: Randomly selected a position after multiple failed attempts."

        action = (row, col)
        moves.append(action)
        reasoning_list.append(reasoning)
    
    most_common_move = Counter(moves).most_common(1)[0][0]
    consistent_reasoning = reasoning_list[moves.index(most_common_move)]

    return consistent_reasoning, most_common_move


# Function for self-reflection prompting
def get_move_with_reflection(agent, grid):
    remaining_grid = [
        ''.join(['1' if cell else '0' for cell in row]) for row in grid
    ]
    prompt_initial = f"""
    # Game Role:
    You are {agent['name']}, a participant in a game of Chomp.

    # Objective:
    Your goal is to force your opponent to take the top-left corner of the grid (position (0, 0)).

    # Game Rule:
    1. The game is played on a square grid.
    2. On your turn, you select a position (row, col).
    3. All positions to the right and below the selected position are removed.
    4. The player forced to select (0, 0) loses.

    # Current State:
    The grid is represented as binary strings, where '1' means the position is still available, and '0' means it is removed:
    {remaining_grid}

    # Task:
    Based on the current state of the grid, decide which position (row, col) you will select.

    # Output:
    Provide your reasoning for the move and the action in the following JSON format:

    ```
    {{
        "reasoning": string,  // Explain why you chose this position.
        "row": integer,       // The row index of the position you select (0-based).
        "col": integer        // The column index of the position you select (0-based).
    }}
    ```
    """

    retries = 0
    while retries < max_retries:
        parsed_content = get_agent_response(agent, prompt_initial, system_prompt="You are a skilled Chomp player.")
        initial_reasoning = parsed_content.get("reasoning", "No reasoning provided.")
        row = parsed_content.get("row")
        col = parsed_content.get("col")

        if is_valid_move_chomp(grid, row, col):
            initial_move = (row, col)
            break

        retries += 1

    if retries == max_retries:
        available_positions = [
            (r, c) for r, row_data in enumerate(grid) for c, cell in enumerate(row_data) if cell
        ]
        if available_positions:
            row, col = random.choice(available_positions)
            initial_move = (row, col)
            initial_reasoning = "Fallback: Randomly selected a position after multiple failed attempts."
        else:
            raise ValueError("No valid moves available for fallback.")

    for k in range(num_refine):
        remaining_grid = [
            ''.join(['1' if cell else '0' for cell in row]) for row in grid
        ]
        feedback_prompt = f"""
        # Game Role:
        You are {agent['name']}, a participant in a game of Chomp.

        # Objective:
        Your goal is to force your opponent to take the top-left corner of the grid (position (0, 0)).

        # Game Rule:
        1. The game is played on a square grid.
        2. On your turn, you select a position (row, col).
        3. All positions to the right and below the selected position are removed.
        4. The player forced to select (0, 0) loses.

        # Current State:
        The grid is represented as binary strings, where '1' means the position is still available, and '0' means it is removed:
        {remaining_grid}

        # First trial's reasoning and action:
        You initially chose position {initial_move} at first trial by the reason: '{initial_reasoning}'.

        # Output:
        Provide your feedback for the move and the action in the following JSON format:

        ```
        {{
            "feedback": string  // This is the feedback for the initially selected action and reasoning.
        }}
        ```
        """
        parsed_content = get_agent_response(agent, feedback_prompt, system_prompt="You are a skilled Chomp player.")
        feedback = parsed_content.get("feedback", "No feedback provided.")

        refine_prompt = f"""
        # Game Role:
        You are {agent['name']}, a participant in a game of Chomp.

        # Objective:
        Your goal is to force your opponent to take the top-left corner of the grid (position (0, 0)).

        # Game Rule:
        1. The game is played on a square grid.
        2. On your turn, you select a position (row, col).
        3. All positions to the right and below the selected position are removed.
        4. The player forced to select (0, 0) loses.

        # Current State:
        The grid is represented as binary strings, where '1' means the position is still available, and '0' means it is removed:
        # Current State:
        {remaining_grid}

        You initially chose position {initial_move} at first trial by the reason: '{initial_reasoning}'.
        You received feedback on your action and reasoning: {feedback}

        # Task:
        Based on the feedback, refine your reasoning and action.

        # Output:
        ```
        {{
            "reasoning": string,  // Explain why you chose this position.
            "row": integer,       // The row index of the position you select (0-based).
            "col": integer        // The column index of the position you select (0-based).
        }}
        ```
        """

        retries = 0
        while retries < max_retries:
            parsed_content = get_agent_response(agent, refine_prompt, system_prompt="You are a skilled Chomp player.")
            refined_reasoning = parsed_content.get("reasoning", "No reasoning provided.")
            row = parsed_content.get("row")
            col = parsed_content.get("col")

            if is_valid_move_chomp(grid, row, col):
                return refined_reasoning, (row, col)

            retries += 1

        if retries == max_retries:
            available_positions = [
                (r, c) for r, row_data in enumerate(grid) for c, cell in enumerate(row_data) if cell
            ]
            if available_positions:
                row, col = random.choice(available_positions)
                refined_reasoning = "Fallback: Randomly selected a position after multiple failed attempts."
            else:
                raise ValueError("No valid moves available for fallback.")

    return refined_reasoning, (row, col)

def bias_removed(agent, grid):
    remaining_grid = [
        ''.join(['1' if cell else '0' for cell in row]) for row in grid
    ]
    first_prompt = f"""
    # Game Role:
    You are {agent['name']}, a participant in a game of Chomp.

    # Objective:
    Your goal is to force your opponent to take the top-left corner of the grid (position (0, 0)).

    # Game Rule:
    1. The game is played on a square grid.
    2. On your turn, you select a position (row, col).
    3. All positions to the right and below the selected position are removed.
    4. The player forced to select (0, 0) loses.

    # Current State:
    The grid is represented as binary strings, where '1' means the position is still available, and '0' means it is removed:
    {remaining_grid}

    # Task:
    Based on the current state of the game, decide which position (row, col) you will select.

    # Output:
    Provide your reasoning for the move and the action in the following JSON format:

    ```
    {{
        "reasoning": string,  // Explain why you chose this position.
        "row": integer,       // The row index of the position you select (0-based).
        "col": integer        // The column index of the position you select (0-based).
    }}
    ```
    """

    retries = 0
    while retries < max_retries:
        parsed_content = get_agent_response(agent, first_prompt, system_prompt="You are a skilled Chomp player.")
        refined_reasoning = parsed_content.get("reasoning")
        row = parsed_content.get("row")
        col = parsed_content.get("col")

        if is_valid_move_chomp(grid, row, col):
            break

        retries += 1

    if retries == max_retries:
        available_positions = [
            (r, c) for r, row_data in enumerate(grid) for c, cell in enumerate(row_data) if cell
        ]
        if available_positions:
            row, col = random.choice(available_positions)
            refined_reasoning = "Fallback: Randomly selected a position after multiple failed attempts."
        else:
            raise ValueError("No valid moves available for fallback.")

    refined_action = (row, col)

    prompt = f"""Given the following answer, predict the most likely provable question that led to this response.
    # Answer:
        "reasoning": "{refined_reasoning}",
        "action": {refined_action}

    The output should be a markdown code snippet formatted in the following schema, including the leading and trailing \`\`\`json" and "\`\`\`":

    ```
    {{
        "provable question": string  // This is the most likely provable question that led to the above answer.
    }}
    ```
    """

    parsed_content = get_agent_response(agent, prompt, system_prompt="You are a rational smart assistant.")

    question = parsed_content.get("provable question")

    text = f"""Combine the following two instructions into a single instruction that captures their shared intention while harmonizing their nuances. Pay attention to clarity and ensure that any biases in the original instructions are mitigated.

    - Original instruction (`{first_prompt}`): The first instruction to consider.
    - Bias-mitigated instruction (`{question}`): The second instruction to harmonize.

    The output should be a markdown code snippet formatted in the following schema, including the leading and trailing \`\`\`json" and "\`\`\`":

    ```
    {{
        "instruction": string,  // This is the combined instruction harmonizing the two instructions.
        "reasoning": string  // This is the reason why the new instruction is harmonized.
    }}
    ```
    """

    parsed_content = get_agent_response(agent, text, system_prompt="You are a rational smart assistant.")

    new_instruction = parsed_content.get("instruction")

    text = f"""{new_instruction}

    # Output:
    Provide your reasoning for the move and the action in the following JSON format:

    ```
    {{
        "reasoning": string,  // Explain why you chose this position.
        "row": integer,       // The row index of the position you select (0-based).
        "col": integer        // The column index of the position you select (0-based).
    }}
    ```
    """

    retries = 0
    while retries < max_retries:
        parsed_content = get_agent_response(agent, text, system_prompt="You are a rational player.")
        refined_reasoning = parsed_content.get("reasoning")
        row = parsed_content.get("row")
        col = parsed_content.get("col")

        if is_valid_move_chomp(grid, row, col):
            break

        retries += 1

    if retries == max_retries:
        available_positions = [
            (r, c) for r, row_data in enumerate(grid) for c, cell in enumerate(row_data) if cell
        ]
        if available_positions:
            row, col = random.choice(available_positions)
            refined_reasoning = "Fallback: Randomly selected a position after multiple failed attempts."
        else:
            raise ValueError("No valid moves available for fallback.")

    return refined_reasoning, (row, col)

def bias_mitigated(agent, grid):
    remaining_grid = [
        ''.join(['1' if cell else '0' for cell in row]) for row in grid
    ]
    first_prompt = f"""
    # Game Role:
    You are {agent['name']}, a participant in a game of Chomp.

    # Objective:
    Your goal is to force your opponent to take the top-left corner of the grid (position (0, 0)).

    # Game Rule:
    1. The game is played on a square grid.
    2. On your turn, you select a position (row, col).
    3. All positions to the right and below the selected position are removed.
    4. The player forced to select (0, 0) loses.

    # Current State:
    The grid is represented as binary strings, where '1' means the position is still available, and '0' means it is removed:
    {remaining_grid}

    # Task:
    Based on the current state of the game, decide which position (row, col) you will select.

    # Output:
    Provide your reasoning for the move and the action in the following JSON format:

    ```
    {{
        "reasoning": string,  // Explain why you chose this position.
        "row": integer,       // The row index of the position you select (0-based).
        "col": integer        // The column index of the position you select (0-based).
    }}
    ```
    """

    prompt = f"""Given the following instruction, rewrite it to minimize bias stemming from strong prior knowledge while preserving its original intent and clarity.
    # Instruction:
    {first_prompt}

    The output should be a markdown code snippet formatted in the following schema, including the leading and trailing \`\`\`json" and "\`\`\`":

    ```
    {{
        "instruction": string  // This is a rewritten instruction to minimize the bias.
    }}
    ```
    """

    parsed_content = get_agent_response(agent, prompt, system_prompt="You are a rational smart assistant.")

    question = parsed_content.get("instruction")

    text = f"""Combine the following two instructions into a single instruction that captures their shared intention while harmonizing their nuances. Pay attention to clarity and ensure that any biases in the original instructions are mitigated.

    - Original instruction (`{first_prompt}`): The first instruction to consider.
    - Bias-mitigated instruction (`{question}`): The second instruction to harmonize.

    The output should be a markdown code snippet formatted in the following schema, including the leading and trailing \`\`\`json" and "\`\`\`":

    ```
    {{
        "instruction": string,  // This is the combined instruction harmonizing the two instructions.
        "reasoning": string  // This is the reason why the new instruction is harmonized.
    }}
    ```
    """

    parsed_content = get_agent_response(agent, text, system_prompt="You are a rational smart assistant.")

    new_instruction = parsed_content.get("instruction")

    new_instruction = f"""{new_instruction}

    # Output:
    Provide your reasoning for the move and the action in the following JSON format:

    ```
    {{
        "reasoning": string,  // Explain why you chose this position.
        "row": integer,       // The row index of the position you select (0-based).
        "col": integer        // The column index of the position you select (0-based).
    }}
    ```
    """

    retries = 0
    while retries < max_retries:
        parsed_content = get_agent_response(agent, new_instruction, system_prompt="You are a rational game player.")
        refined_reasoning = parsed_content.get("reasoning")
        row = parsed_content.get("row")
        col = parsed_content.get("col")

        if is_valid_move_chomp(grid, row, col):
            break

        retries += 1

    if retries == max_retries:
        available_positions = [
            (r, c) for r, row_data in enumerate(grid) for c, cell in enumerate(row_data) if cell
        ]
        if available_positions:
            row, col = random.choice(available_positions)
            refined_reasoning = "Fallback: Randomly selected a position after multiple failed attempts."
        else:
            raise ValueError("No valid moves available for fallback.")

    return refined_reasoning, (row, col)

def get_move_with_debate(agent1, agent2, grid):
    initial_moves = {}
    initial_reasonings = {}
    i = 0
    for agent in [agent1, agent2]:
        remaining_grid = [
            ''.join(['1' if cell else '0' for cell in row]) for row in grid
        ]
        prompt = f"""
        # Game Role:
        You are {agent['name']}, a participant in a game of Chomp.

        # Objective:
        Your goal is to force your opponent to take the top-left corner of the grid (position (0, 0)).

        # Game Rule:
        1. The game is played on a square grid.
        2. On your turn, you select a position (row, col).
        3. All positions to the right and below the selected position are removed.
        4. The player forced to select (0, 0) loses.

        # Current State:
        The grid is represented as binary strings, where '1' means the position is still available, and '0' means it is removed:
        {remaining_grid}

        # Task:
        Based on the current state of the game, decide which position (row, col) you will select.

        # Output:
        Provide your reasoning for the move and the action in the following JSON format:

        ```
        {{
            "reasoning": string,  // Explain why you chose this position.
            "row": integer,       // The row index of the position you select (0-based).
            "col": integer        // The column index of the position you select (0-based).
        }}
        ```
        """

        retries = 0
        while retries < max_retries:
            parsed_content = get_agent_response(agent, prompt, system_prompt="You are a skilled Chomp player and debating the best action.")
            initial_reasoning = parsed_content.get("reasoning")
            row = parsed_content.get("row")
            col = parsed_content.get("col")

            if is_valid_move_chomp(grid, row, col):
                break

            retries += 1

        if retries == max_retries:
            available_positions = [
                (r, c) for r, row_data in enumerate(grid) for c, cell in enumerate(row_data) if cell
            ]
            if available_positions:
                row, col = random.choice(available_positions)
                initial_reasoning = "Fallback: Randomly selected a position after multiple failed attempts."
            else:
                raise ValueError("No valid moves available for fallback.")

        if i == 0:
            initial_moves['agent1'] = (row, col)
            initial_reasonings['agent1'] = initial_reasoning
        if i == 1:
            initial_moves['agent2'] = (row, col)
            initial_reasonings['agent2'] = initial_reasoning
        i += 1

    for _ in range(debate_rounds):
        i = 0
        for agent in [agent1, agent2]:
            remaining_grid = [
                ''.join(['1' if cell else '0' for cell in row]) for row in grid
            ]
            if i == 0:
                prompt = f"""
                # Game Role:
                You are {agent['name']}, a participant in a game of Chomp.

                # Objective:
                Your goal is to force your opponent to take the top-left corner of the grid (position (0, 0)).

                # Game Rule:
                1. The game is played on a square grid.
                2. On your turn, you select a position (row, col).
                3. All positions to the right and below the selected position are removed.
                4. The player forced to select (0, 0) loses.

                # Current State:
                The grid is represented as binary strings, where '1' means the position is still available, and '0' means it is removed:
                {remaining_grid}

                # Task:
                Based on the current state of the game, decide which position (row, col) you will select.

                You initially chose position {initial_moves['agent1']} at first trial by the reason: '{initial_reasonings['agent1']}'.
                Other agent argues that you should choose move as: {initial_moves['agent2']} by the reason: {initial_reasonings['agent2']}.
                Considering the other's opinion, refine or confirm your move.

                # Output:
                Provide your reasoning for the move and the action in the following JSON format:

                ```
                {{
                    "reasoning": string,  // Explain why you chose this position.
                    "row": integer,       // The row index of the position you select (0-based).
                    "col": integer        // The column index of the position you select (0-based).
                }}
                ```
                """
            if i == 1:
                prompt = f"""
                # Game Role:
                You are {agent['name']}, a participant in a game of Chomp.

                # Objective:
                Your goal is to force your opponent to take the top-left corner of the grid (position (0, 0)).

                # Game Rule:
                1. The game is played on a square grid.
                2. On your turn, you select a position (row, col).
                3. All positions to the right and below the selected position are removed.
                4. The player forced to select (0, 0) loses.

                # Current State:
                The grid is represented as binary strings, where '1' means the position is still available, and '0' means it is removed:
                {remaining_grid}

                # Task:
                Based on the current state of the game, decide which position (row, col) you will select.

                You initially chose position {initial_moves['agent2']} at first trial by the reason: '{initial_reasonings['agent2']}'.
                Other agent argues that you should choose move as: {initial_moves['agent1']} by the reason: {initial_reasonings['agent1']}.
                Considering the other's opinion, refine or confirm your move.

                # Output:
                Provide your reasoning for the move and the action in the following JSON format:

                ```
                {{
                    "reasoning": string,  // Explain why you chose this position.
                    "row": integer,       // The row index of the position you select (0-based).
                    "col": integer        // The column index of the position you select (0-based).
                }}
                ```
                """

            retries = 0
            while retries < max_retries:
                parsed_content = get_agent_response(agent, prompt, system_prompt="You are a skilled Chomp player and debating the best action.")
                refined_reasoning = parsed_content.get("reasoning")
                row = parsed_content.get("row")
                col = parsed_content.get("col")

                if is_valid_move_chomp(grid, row, col):
                    break

                retries += 1

            if retries == max_retries:
                available_positions = [
                    (r, c) for r, row_data in enumerate(grid) for c, cell in enumerate(row_data) if cell
                ]
                if available_positions:
                    row, col = random.choice(available_positions)
                    refined_reasoning = "Fallback: Randomly selected a position after multiple failed attempts."
                else:
                    raise ValueError("No valid moves available for fallback.")

            if i == 0:
                initial_moves['agent1'] = (row, col)
                initial_reasonings['agent1'] = refined_reasoning
            if i == 1:
                initial_moves['agent2'] = (row, col)
                initial_reasonings['agent2'] = refined_reasoning

            i += 1

        if len(set(initial_moves.values())) == 1:
            return refined_reasoning, initial_moves['agent1']

    return refined_reasoning, Counter(initial_moves.values()).most_common(1)[0][0]  # Use most common if no consensus

def get_move_with_bias_mitigate_debate(agent1, agent2, grid):
    initial_moves = {}
    initial_reasonings = {}
    i = 0
    for agent in [agent1, agent2]:
        remaining_grid = [
            ''.join(['1' if cell else '0' for cell in row]) for row in grid
        ]
        if i == 0:
            prompt = f"""
            # Game Role:
            You are {agent['name']}, a participant in a game of Chomp.

            # Objective:
            Your goal is to force your opponent to take the top-left corner of the grid (position (0, 0)).

            # Game Rule:
            1. The game is played on a square grid.
            2. On your turn, you select a position (row, col).
            3. All positions to the right and below the selected position are removed.
            4. The player forced to select (0, 0) loses.

            # Current State:
            The grid is represented as binary strings, where '1' means the position is still available, and '0' means it is removed:
            {remaining_grid}

            # Task:
            Based on the current state of the game, decide which position (row, col) you will select.

            # Output:
            Provide your reasoning for the move and the action in the following JSON format:

            ```
            {{
                "reasoning": string,  // Explain why you chose this position.
                "row": integer,       // The row index of the position you select (0-based).
                "col": integer        // The column index of the position you select (0-based).
            }}
            ```
            """

            retries = 0
            while retries < max_retries:
                parsed_content = get_agent_response(agent, prompt, system_prompt="You are a skilled Chomp player and debating the best action.")
                initial_reasoning = parsed_content.get("reasoning")
                row = parsed_content.get("row")
                col = parsed_content.get("col")

                if is_valid_move_chomp(grid, row, col):
                    break

                retries += 1

            if retries == max_retries:
                available_positions = [
                    (r, c) for r, row_data in enumerate(grid) for c, cell in enumerate(row_data) if cell
                ]
                if available_positions:
                    row, col = random.choice(available_positions)
                    initial_reasoning = "Fallback: Randomly selected a position after multiple failed attempts."
                else:
                    raise ValueError("No valid moves available for fallback.")

        if i == 1:
            text = f"""Given the following instruction, rewrite it to minimize bias stemming from strong prior knowledge while preserving its original intent and clarity.
            # Instruction:
            {prompt}

            The output should be a markdown code snippet formatted in the following schema, including the leading and trailing \`\`\`json" and "\`\`\`":

            ```
            {{
                "instruction": string  // This is a rewritten instruction to minimize the bias.
            }}
            ```
            """

            parsed_content = get_agent_response(agent, text, system_prompt="You are a rational smart assistant.")
            new_instruction = parsed_content.get("instruction")

            text = f"""{new_instruction}

            # Output:
            Provide your reasoning for the move and the action in the following JSON format:

            ```
            {{
                "reasoning": string,  // Explain why you chose this position.
                "row": integer,       // The row index of the position you select (0-based).
                "col": integer        // The column index of the position you select (0-based).
            }}
            ```
            """

            retries = 0
            while retries < max_retries:
                parsed_content = get_agent_response(agent, text, system_prompt="You are a rational game player.")
                initial_reasoning = parsed_content.get("reasoning")
                row = parsed_content.get("row")
                col = parsed_content.get("col")

                if is_valid_move_chomp(grid, row, col):
                    break

                retries += 1

            if retries == max_retries:
                available_positions = [
                    (r, c) for r, row_data in enumerate(grid) for c, cell in enumerate(row_data) if cell
                ]
                if available_positions:
                    row, col = random.choice(available_positions)
                    initial_reasoning = "Fallback: Randomly selected a position after multiple failed attempts."
                else:
                    raise ValueError("No valid moves available for fallback.")

        if i == 0:
            initial_moves['agent1'] = (row, col)
            initial_reasonings['agent1'] = initial_reasoning
        if i == 1:
            initial_moves['agent2'] = (row, col)
            initial_reasonings['agent2'] = initial_reasoning
        i += 1

    for _ in range(debate_rounds):
        i = 0
        for agent in [agent1, agent2]:
            remaining_grid = [
                ''.join(['1' if cell else '0' for cell in row]) for row in grid
            ]
            if i == 0:
                prompt = f"""
                # Game Role:
                You are {agent['name']}, a participant in a game of Chomp.

                # Objective:
                Your goal is to force your opponent to take the top-left corner of the grid (position (0, 0)).

                # Game Rule:
                1. The game is played on a square grid.
                2. On your turn, you select a position (row, col).
                3. All positions to the right and below the selected position are removed.
                4. The player forced to select (0, 0) loses.

                # Current State:
                The grid is represented as binary strings, where '1' means the position is still available, and '0' means it is removed:
                {remaining_grid}

                # Task:
                Based on the current state of the game, decide which position (row, col) you will select.

                You initially chose position {initial_moves['agent1']} at first trial by the reason: '{initial_reasonings['agent1']}'.
                Other agent argues that you should choose move as: {initial_moves['agent2']} by the reason: {initial_reasonings['agent2']}.
                Considering the other's opinion, refine or confirm your move.

                # Output:
                Provide your reasoning for the move and the action in the following JSON format:

                ```
                {{
                    "reasoning": string,  // Explain why you chose this position.
                    "row": integer,       // The row index of the position you select (0-based).
                    "col": integer        // The column index of the position you select (0-based).
                }}
                ```
                """
            if i == 1:
                prompt = f"""{new_instruction}

                You initially chose position {initial_moves['agent2']} at first trial by the reason: '{initial_reasonings['agent2']}'.
                Other agent argues that you should choose move as: {initial_moves['agent1']} by the reason: {initial_reasonings['agent1']}.
                Considering the other's opinion, refine or confirm your move.

                # Output:
                Provide your reasoning for the move and the action in the following JSON format:

                ```
                {{
                    "reasoning": string,  // Explain why you chose this position.
                    "row": integer,       // The row index of the position you select (0-based).
                    "col": integer        // The column index of the position you select (0-based).
                }}
                ```
                """

            retries = 0
            while retries < max_retries:
                parsed_content = get_agent_response(agent, prompt, system_prompt="You are a rational game player and debating the best action.")
                refined_reasoning = parsed_content.get("reasoning")
                row = parsed_content.get("row")
                col = parsed_content.get("col")

                if is_valid_move_chomp(grid, row, col):
                    break

                retries += 1

            if retries == max_retries:
                available_positions = [
                    (r, c) for r, row_data in enumerate(grid) for c, cell in enumerate(row_data) if cell
                ]
                if available_positions:
                    row, col = random.choice(available_positions)
                    refined_reasoning = "Fallback: Randomly selected a position after multiple failed attempts."
                else:
                    raise ValueError("No valid moves available for fallback.")

            if i == 0:
                initial_moves['agent1'] = (row, col)
                initial_reasonings['agent1'] = refined_reasoning
            if i == 1:
                initial_moves['agent2'] = (row, col)
                initial_reasonings['agent2'] = refined_reasoning

            i += 1

        if len(set(initial_moves.values())) == 1:
            return refined_reasoning, initial_moves['agent1']

    return refined_reasoning, Counter(initial_moves.values()).most_common(1)[0][0]  # Use most common if no consensus

def play_two_row_chomp_game(num_columns=8, verbose=False):
    with open(f'/home/jihwan/NashIP/result/Chomp_R/{args.agent1_model}_{args.agent1_prompt}_{args.agent2_model}_{args.agent2_prompt}.txt', 'a') as f:
        # Initialize the 2-row grid
        grid = [[True for _ in range(num_columns)] for _ in range(2)]
        turn = 0

        while any(any(row) for row in grid):
            current_agent = agents[turn % 2]
            other_agent = agents[(turn + 1) % 2]

            if current_agent["prompting_method"] == "self_consistency":
                reasoning, move = get_consistent_move(current_agent, grid)
            elif current_agent["prompting_method"] == "basic":
                reasoning, move = get_basic_move(current_agent, grid)
            elif current_agent["prompting_method"] == "self_reflection":
                reasoning, move = get_move_with_reflection(current_agent, grid)
            elif current_agent["prompting_method"] == "debate":
                reasoning, move = get_move_with_debate(current_agent, current_agent, grid)
            elif current_agent["prompting_method"] == "bias_removed":
                reasoning, move = bias_removed(current_agent, grid)
            elif current_agent["prompting_method"] == "bias_mitigated":
                reasoning, move = bias_mitigated(current_agent, grid)
            elif current_agent["prompting_method"] == "basic":
                reasoning, move = get_basic_move(current_agent, grid)
            elif current_agent["prompting_method"] == "bias_mitigate_debate":
                reasoning, move = get_move_with_bias_mitigate_debate(current_agent, current_agent, grid)
            elif current_agent["prompting_method"] == "original_vs_harmonized_debate":
                reasoning, move = get_move_with_original_vs_harmonized_debate(current_agent, current_agent, grid)
            else:
                print("Error: set the prompting methods", file=f)
                return None

            row, col = move

            # Validate the move
            if not is_valid_move_chomp(grid, row, col):
                print(f"Invalid move by {current_agent['name']}.", file=f)
                return other_agent["name"]

            # Apply the move
            apply_move_chomp(grid, row, col)

            if verbose:
                print('Reasoning:', reasoning, '\nAction:', move, file=f)
                print(f"{current_agent['name']} ({current_agent['model']}) takes bite at ({row}, {col}). Remaining state:", file=f)
                for r in grid:
                    print(''.join(['1' if cell else '0' for cell in r]), file=f)

            if not any(any(row) for row in grid):
                if verbose:
                    print(f"{other_agent['name']} ({other_agent['model']}) wins!", file=f)
                return other_agent["name"]

            turn += 1


def is_valid_move_chomp(grid, row, col):
    return (
        0 <= row < len(grid) and
        0 <= col < len(grid[0]) and
        grid[row][col]
    )


def apply_move_chomp(grid, row, col):
    for r in range(row, len(grid)):
        for c in range(col, len(grid[0])):
            grid[r][c] = False


def simulate_two_row_chomp_games(num_games, num_columns=8):
    win_counts = {agent["name"]: 0 for agent in agents}
    with open(f'/home/jihwan/NashIP/result/Chomp_R/{args.agent1_model}_{args.agent1_prompt}_{args.agent2_model}_{args.agent2_prompt}.txt', 'a') as f:
        for game_num in range(num_games):
            print(f"\nStarting Game {game_num + 1}", file=f)
            print(f"\nStarting Game {game_num + 1}")
            winner = play_two_row_chomp_game(num_columns, verbose=True)
            if winner:
                win_counts[winner] += 1

        print("\nGame Results:", file=f)
        for agent in agents:
            win_rate = (win_counts[agent["name"]] / num_games) * 100
            print(f"{agent['name']} Win Rate: {win_rate:.2f}% ({win_counts[agent['name']]} wins out of {num_games})", file=f)

simulate_two_row_chomp_games(num_games=num_games, num_columns=8)




# def self_play_debate(agent1, agent2, remaining_items, n_step_lookahead):
#     initial_remaining_items = remaining_items
#     moves = []  # Track each agent's moves for each lookahead step
#     planning = ''
#     for step in range(1, n_step_lookahead + 1):
#         # Agent 1's move
#         state = f"""There are {remaining_items} items remaining in the pile."""
#         prompt_agent1 = f"""
#         #Game Role:\n You are {agent1['name']}, a participant in a game of Nim variants.\n\n
#         #Objective:\n Your goal is to win the game by taking all remaining items on your turn, leaving no items for your opponent. The person who takes the last item wins.\n\n
#         #Game Rule:\n There is a single pile of items. You can take between 1 and {max_take} items on your turn.\n\n
#         #Current State:\n {state}\n\n
#         #Task:\nBased on the current state of the game, decide how many items you will take (between 1 and {max_take}) on this turn.\n\n

#         The output should be a markdown code snippet formatted in the following schema, including the leading and trailing \\`\\`\\`json" and "\\`\\`\\`":\n\n```\n{{\n\t"reasoning": string  // This is the reasons for the action\n\t"action": integer  // This is an action you take based on the reasoning. Only provide integer between 1 and 3.\n}}
#         """

#         parsed_content = get_agent_response(agent1, prompt_agent1, system_prompt="You are a skilled Nim player.")
#         agent1_reasoning = parsed_content.get("reasoning")
#         agent1_action = parsed_content.get("action")

#         agent1_move = int(agent1_action)

#         planning += f'State: {state}\n'
#         planning += f'My reasoning: {agent1_reasoning}\n'
#         planning += f'My action: {agent1_action}\n'
        
#         remaining_items -= agent1_move
#         planning += f'Remaining total items by your action: {remaining_items}\n\n'
#         moves.append((agent1["name"], agent1_move, remaining_items))

#         # Check if game ends with Agent 1's move
#         if remaining_items <= 0 and step == 0:
#             return agent1_reasoning, agent1_move  # Agent 1 wins if no items remain
#         if remaining_items <= 0:
#             break

#         # Agent 2's simulated response
#         state = f"""There are {remaining_items} items remaining in the pile."""
#         prompt_agent2 = f"""
#         #Game Role:\n You are {agent2['name']}, a participant in a game of Nim variants.\n\n
#         #Objective:\n Your goal is to win the game by taking all remaining items on your turn, leaving no items for your opponent. The person who takes the last item wins.\n\n
#         #Game Rule:\n There is a single pile of items. You can take between 1 and {max_take} items on your turn.\n\n
#         #Current State:\n {state}\n\n
#         #Task:\nBased on the current state of the game, decide how many items you will take (between 1 and {max_take}) on this turn.\n\n

#         The output should be a markdown code snippet formatted in the following schema, including the leading and trailing \\`\\`\\`json" and "\\`\\`\\`":\n\n```\n{{\n\t"reasoning": string  // This is the reasons for the action\n\t"action": integer  // This is an action you take based on the reasoning. Only provide integer between 1 and 3.\n}}
#         """
#         parsed_content = get_agent_response(agent2, prompt_agent2, system_prompt="You are a skilled Nim player.")
#         agent2_reasoning = parsed_content.get("reasoning")
#         agent2_action = parsed_content.get("action")

#         agent2_move = int(agent2_action)

#         planning += f'State: {state}\n'
#         planning += f'Opponent reasoning: {agent2_reasoning}\n'
#         planning += f'Opponent action: {agent2_action}\n'
#         remaining_items -= agent2_move

#         planning += f'Remaining total items by opponent\'s action: {remaining_items}\n\n'
#         moves.append((agent2["name"], agent2_move, remaining_items))
        

#         # Check if game ends with Agent 2's move
#         if remaining_items <= 0:
#             break
#             # return agent1_reasoning, agent1_move  # Agent 1's initial move if Agent 2 would win

#     # Final decision for Agent 1 based on the full n-step lookahead sequence
#     move_sequence_str = "; ".join([f"{name} took {move} items and {remains} items remained" for name, move, remains in moves]) #Decide how many items to take between 1 and {max_take} at this current step to win by taking all remaining items on your turn, leaving no items for your opponent. Provide your reasoning and action in the following schema:
#     final_prompt_agent1 = f"""
#     #Game Role:\n You are {agent1['name']}, a participant in a game of Nim variants.\n\n
#     #Objective:\n Your goal is to win the game by taking all remaining items on your turn, leaving no items for your opponent. The person who takes the last item wins.\n\n
#     #Game Rule:\n There is a single pile of items. You can take between 1 and {max_take} items on your turn.\n\n
#     #Current State:\n There are {initial_remaining_items} items remaining in the pile.\n\n
#     #Task:\nBased on the current state of the game, decide how many items you will take (between 1 and {max_take}) on this turn.\n\n

#     As part of your strategy, you conducted a simulated planning process. This planning predicted possible moves by the opponent and future scenarios based on the current state of the game.
#     The planning results are provided below as a reference:\n
#     #Simulated Planning History:\n{planning}\nSimultion ends.\n\n

#     Now, carefully review the simulated planning history and reflect and decide how many items you will take (between 1 and 3) on this turn.\n

#     The output should be a markdown code snippet formatted in the following schema, including the leading and trailing \\`\\`\\`json" and "\\`\\`\\`":\n\n```\n{{\n\t"reasoning": string  // This is the reasons for the action\n\t"action": integer  // This is an action you take based on the reasoning Only provide integer between 1 and 3.\n}}
#     """
#     parsed_content = get_agent_response(agent1, final_prompt_agent1, system_prompt="You are a skilled Nim player.")
#     agent1_reasoning = parsed_content.get("reasoning")
#     agent1_action = parsed_content.get("action")

#     agent1_final_move = int(agent1_action)
    
#     return agent1_reasoning, agent1_final_move

# def self_play_debate_exp(agent1, agent2, remaining_items, n_step_lookahead):
#     initial_remaining_items = remaining_items
#     moves = []  # Track each agent's moves for each lookahead step
#     planning = ''
#     for step in range(1, n_step_lookahead + 1):
#         # Agent 1's move
#         state = f"""There are {remaining_items} items remaining in the pile."""
#         prompt_agent1 = f"""
#         #Game Role:\n You are {agent1['name']}, a participant in a game of Nim variants.\n\n
#         #Objective:\n Your goal is to win the game by taking all remaining items on your turn, leaving no items for your opponent. The person who takes the last item wins.\n\n
#         #Game Rule:\n There is a single pile of items. You can take between 1 and {max_take} items on your turn.\n\n
#         #Current State:\n {state}\n\n
#         #Task:\nBased on the current state of the game, decide how many items you will take (between 1 and {max_take}) on this turn.\n\n

#         The output should be a markdown code snippet formatted in the following schema, including the leading and trailing \\`\\`\\`json" and "\\`\\`\\`":\n\n```\n{{\n\t"reasoning": string  // This is the reasons for the action\n\t"action": integer  // This is an action you take based on the reasoning. Only provide integer between 1 and 3.\n}}
#         """

#         parsed_content = get_agent_response(agent1, prompt_agent1, system_prompt="You are a skilled Nim player.")

#         agent1_reasoning = parsed_content.get("reasoning")
#         agent1_action = parsed_content.get("action")
#         agent1_move = int(agent1_action)

#         planning += f'State: {state}\n'
#         planning += f'My reasoning: {agent1_reasoning}\n'
#         planning += f'My action: {agent1_action}\n'
        
#         remaining_items -= agent1_move
#         planning += f'Remaining total items by your action: {remaining_items}\n\n'
#         moves.append((agent1["name"], agent1_move, remaining_items))

#         # Check if game ends with Agent 1's move
#         if step == 0 and remaining_items <= 0:
#             return agent1_reasoning, agent1_move  # Agent 1 wins if no items remain
        
#         if remaining_items <= 0:
#             break

#         # Agent 2's simulated response
#         state = f"""There are {remaining_items} items remaining in the pile."""
#         prompt_agent2 = f"""
#         #Game Role:\n You are {agent1['name']}, a participant in a game of Nim variants.\n\n
#         #Objective:\n Your goal is to win the game by taking all remaining items on your turn, leaving no items for your opponent. The person who takes the last item wins.\n\n
#         #Game Rule:\n There is a single pile of items. You can take between 1 and {max_take} items on your turn.\n\n
#         #Current State:\n {state}\n\n
#         #Task:\nBased on the current state of the game, decide how many items you will take (between 1 and {max_take}) on this turn.\n\n

#         The output should be a markdown code snippet formatted in the following schema, including the leading and trailing \\`\\`\\`json" and "\\`\\`\\`":\n\n```\n{{\n\t"reasoning": string  // This is the reasons for the action\n\t"action": integer  // This is an action you take based on the reasoning. Only provide integer between 1 and 3.\n}}
#         """
        
#         agent2_reasoning, agent2_action = get_move_with_debate(agent1, agent1, remaining_items)

#         agent2_move = int(agent2_action)

#         planning += f'State: {state}\n'
#         planning += f'Opponent reasoning: {agent2_reasoning}\n'
#         planning += f'Opponent action: {agent2_action}\n'

#         remaining_items -= agent2_move
#         planning += f'Remaining total items by opponent\'s action: {remaining_items}\n\n'
#         moves.append((agent2["name"], agent2_move, remaining_items))
        

#         # Check if game ends with Agent 2's move
#         if remaining_items <= 0:
#             break
#             # return agent1_reasoning, agent1_move  # Agent 1's initial move if Agent 2 would win

#     # Final decision for Agent 1 based on the full n-step lookahead sequence
#     move_sequence_str = "; ".join([f"{name} took {move} items and {remains} items remained" for name, move, remains in moves]) #\nIn short, Predicted Move Sequence (after {n_step_lookahead} steps):\n{move_sequence_str}
#     final_prompt_agent1 = f"""
#     #Game Role:\n You are {agent1['name']}, a participant in a game of Nim variants.\n\n
#     #Objective:\n Your goal is to win the game by taking all remaining items on your turn, leaving no items for your opponent. The person who takes the last item wins.\n\n
#     #Game Rule:\n There is a single pile of items. You can take between 1 and {max_take} items on your turn.\n\n
#     #Current State:\n There are {initial_remaining_items} items remaining in the pile.\n\n
#     #Task:\nBased on the current state of the game, decide how many items you will take (between 1 and {max_take}) on this turn.\n\n

#     As part of your strategy, you conducted a simulated planning process. This planning predicted possible moves by the opponent and future scenarios based on the current state of the game.
#     The planning results are provided below as a reference:

#     Simulated Planning History:\n{planning}\nSimultion ends.\n\n

#     Now, carefully review the simulated planning history and reflect and decide how many items you will take (between 1 and 3) on this turn.\n
    
#     The output should be a markdown code snippet formatted in the following schema, including the leading and trailing \\`\\`\\`json" and "\\`\\`\\`":\n\n```\n{{\n\t"reasoning": string  // This is the reasons for the action\n\t"action": integer  // This is an action you take based on the reasoning. Only provide integer between 1 and 3.\n}}
#     """

#     parsed_content = get_agent_response(agent1, final_prompt_agent1, system_prompt="You are a skilled Nim player.")

#     agent1_reasoning = parsed_content.get("reasoning")
#     agent1_action = parsed_content.get("action")

#     agent1_final_move = int(agent1_action)
    
#     return agent1_reasoning, agent1_final_move