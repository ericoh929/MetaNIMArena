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
gemini_api_keys = [
    'AIzaSyB5lNkfPiDJMkwNrpAv3MHbnnFvkpuF7Ok',  # 첫 번째 API 키
    'AIzaSyCYkix3fQio-WpUus7ziwNGhSk6qZp7LJs'   # 두 번째 API 키
]
selected_api_key = random.choice(gemini_api_keys)
genai.configure(api_key=selected_api_key)

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

def get_agent_response(agent, prompt, system_prompt="You are a skilled Nim player.", temperature=0.7):
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

            elif agent["model"] in ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"]:
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


agents = [
    {"name": "Agent 1", "model": args.agent1_model, "prompting_method": args.agent1_prompt},
    {"name": "Agent 2", "model": args.agent2_model, "prompting_method": args.agent2_prompt}
]

def get_move(agent, grid):
    remaining_grid = [
        ''.join(['1' if cell else '0' for cell in row]) for row in grid
    ]
    prompt = f"""
    # Game Role:\nYou are {agent['name']}, a participant in a game of NxN square Chomp.\n
    # Objective:\nYour goal is to force your opponent to take the top-right corner of the grid (position (N-1, N-1)).\n
    # Game Rule:\n1. The game is played on a square grid.\n2. On your turn, you select a position (row, col).\n3. All positions to the left and below the selected position are removed.\n4. The player forced to select (N-1, N-1) loses.\n
    # Coordinate System:\n- The grid follows a **zero-based coordinate system**.\n- The **bottom-left corner** is (0, 0).\n
    # Current State:\nThe grid is represented as a list of lists, where '1' means the position is still available, and '0' means it is removed:\n{remaining_grid}\n
    # Task:\nBased on the current state of the grid, decide which position (row, col) you will select.",

    # Output:
    Provide the action in the following JSON format:

    ```
    {{
        "row": integer,       // The row index of the position you select (0-based).
        "col": integer        // The column index of the position you select (0-based).
    }}
    ```
    """

    retries = 0
    while retries < max_retries:
        parsed_content = get_agent_response(agent, prompt, system_prompt="You are a skilled Chomp player.")
        reasoning = "None"
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

# Function for basic move (single-response without consistency or modeling)
def get_basic_move(agent, grid):
    remaining_grid = [
        ''.join(['1' if cell else '0' for cell in row]) for row in grid
    ]
    prompt = f"""
    # Game Role:\nYou are {agent['name']}, a participant in a game of NxN square Chomp.\n
    # Objective:\nYour goal is to force your opponent to take the top-right corner of the grid (position (N-1, N-1)).\n
    # Game Rule:\n1. The game is played on a square grid.\n2. On your turn, you select a position (row, col).\n3. All positions to the left and below the selected position are removed.\n4. The player forced to select (N-1, N-1) loses.\n
    # Coordinate System:\n- The grid follows a **zero-based coordinate system**.\n- The **bottom-left corner** is (0, 0).\n
    # Current State:\nThe grid is represented as a list of lists, where '1' means the position is still available, and '0' means it is removed:\n{remaining_grid}\n
    # Task:\nBased on the current state of the grid, decide which position (row, col) you will select.",

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
    # Game Role:\nYou are {agent['name']}, a participant in a game of NxN square Chomp.\n
    # Objective:\nYour goal is to force your opponent to take the top-right corner of the grid (position (N-1, N-1)).\n
    # Game Rule:\n1. The game is played on a square grid.\n2. On your turn, you select a position (row, col).\n3. All positions to the left and below the selected position are removed.\n4. The player forced to select (N-1, N-1) loses.\n
    # Coordinate System:\n- The grid follows a **zero-based coordinate system**.\n- The **bottom-left corner** is (0, 0).\n
    # Current State:\nThe grid is represented as a list of lists, where '1' means the position is still available, and '0' means it is removed:\n{remaining_grid}\n
    # Task:\nBased on the current state of the grid, decide which position (row, col) you will select.",

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
        # Game Role:\nYou are {agent['name']}, a participant in a game of NxN square Chomp.\n
        # Objective:\nYour goal is to force your opponent to take the top-right corner of the grid (position (N-1, N-1)).\n
        # Game Rule:\n1. The game is played on a square grid.\n2. On your turn, you select a position (row, col).\n3. All positions to the left and below the selected position are removed.\n4. The player forced to select (N-1, N-1) loses.\n
        # Coordinate System:\n- The grid follows a **zero-based coordinate system**.\n- The **bottom-left corner** is (0, 0).\n
        # Current State:\nThe grid is represented as a list of lists, where '1' means the position is still available, and '0' means it is removed:\n{remaining_grid}\n
        # Task:\nBased on the current state of the grid, decide which position (row, col) you will select."

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

def get_move_with_debate(agent1, agent2, grid):
    initial_moves = {}
    initial_reasonings = {}
    i = 0
    for agent in [agent1, agent2]:
        remaining_grid = [
            ''.join(['1' if cell else '0' for cell in row]) for row in grid
        ]
        prompt = f"""
        # Game Role:\nYou are {agent['name']}, a participant in a game of NxN square Chomp.\n
        # Objective:\nYour goal is to force your opponent to take the top-right corner of the grid (position (N-1, N-1)).\n
        # Game Rule:\n1. The game is played on a square grid.\n2. On your turn, you select a position (row, col).\n3. All positions to the left and below the selected position are removed.\n4. The player forced to select (N-1, N-1) loses.\n
        # Coordinate System:\n- The grid follows a **zero-based coordinate system**.\n- The **bottom-left corner** is (0, 0).\n
        # Current State:\nThe grid is represented as a list of lists, where '1' means the position is still available, and '0' means it is removed:\n{remaining_grid}\n
        # Task:\nBased on the current state of the grid, decide which position (row, col) you will select.",

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
                # Game Role:\nYou are {agent['name']}, a participant in a game of NxN square Chomp.\n
                # Objective:\nYour goal is to force your opponent to take the top-right corner of the grid (position (N-1, N-1)).\n
                # Game Rule:\n1. The game is played on a square grid.\n2. On your turn, you select a position (row, col).\n3. All positions to the left and below the selected position are removed.\n4. The player forced to select (N-1, N-1) loses.\n
                # Coordinate System:\n- The grid follows a **zero-based coordinate system**.\n- The **bottom-left corner** is (0, 0).\n
                # Current State:\nThe grid is represented as a list of lists, where '1' means the position is still available, and '0' means it is removed:\n{remaining_grid}\n
                # Task:\nBased on the current state of the grid, decide which position (row, col) you will select.",

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
                # Game Role:\nYou are {agent['name']}, a participant in a game of NxN square Chomp.\n
                # Objective:\nYour goal is to force your opponent to take the top-right corner of the grid (position (N-1, N-1)).\n
                # Game Rule:\n1. The game is played on a square grid.\n2. On your turn, you select a position (row, col).\n3. All positions to the left and below the selected position are removed.\n4. The player forced to select (N-1, N-1) loses.\n
                # Coordinate System:\n- The grid follows a **zero-based coordinate system**.\n- The **bottom-left corner** is (0, 0).\n
                # Current State:\nThe grid is represented as a list of lists, where '1' means the position is still available, and '0' means it is removed:\n{remaining_grid}\n
                # Task:\nBased on the current state of the grid, decide which position (row, col) you will select.",

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

def get_move_dreamad(agent1, agent2, grid):
    initial_moves = {}
    initial_reasonings = {}
    i = 0
    for agent in [agent1]:
        remaining_grid = [
            ''.join(['1' if cell else '0' for cell in row]) for row in grid
        ]
        prompt = f"""
        # Game Role:\nYou are {agent['name']}, a participant in a game of NxN square Chomp.\n
        # Objective:\nYour goal is to force your opponent to take the top-right corner of the grid (position (N-1, N-1)).\n
        # Game Rule:\n1. The game is played on a square grid.\n2. On your turn, you select a position (row, col).\n3. All positions to the left and below the selected position are removed.\n4. The player forced to select (N-1, N-1) loses.\n
        # Coordinate System:\n- The grid follows a **zero-based coordinate system**.\n- The **bottom-left corner** is (0, 0).\n
        # Current State:\nThe grid is represented as a list of lists, where '1' means the position is still available, and '0' means it is removed:\n{remaining_grid}\n
        # Task:\nBased on the current state of the grid, decide which position (row, col) you will select."
        """
        #################prompt1#################
        game_prompt = f"""
        Below is a game description. Extract key information.

        **Game Description:**
        {prompt}

        # Output:
        Provide your answer in the following JSON format:
        ```
        {{
        "game_definition": "string", // What is the definition of this game?.
        "winning_condition": "string", // How to win the game.
        "move_constraints": "string" // What actions are allowed per turn.
        }}
        ```
        """
    
        parsed_content = get_agent_response(agent, game_prompt, system_prompt="You are a game theorist and strategist.",temperature=0.1)

        game_definition = parsed_content.get("game_definition")
        winning_condition = parsed_content.get("winning_condition")
        move_constraints = parsed_content.get("move_constraints")


        strategy_prompt = f"""
        Based on the game information below, derive the **optimal strategy**.

        **Game:** {game_definition}  
        **Winning Condition:** {winning_condition}  
        **Move Constraints:** {move_constraints}

        # Output:
        Provide your answer in the following JSON format:
        ```
        {{
        "state_evaluation": "string", // How to assess the game state.
        "winning_strategy": "string", // Winning strategy in this turn to win this game.
        "endgame_tactics": "string" // Best strategy in a near-win situation.
        }}
        ```
        """
        parsed_content = get_agent_response(agent, strategy_prompt, system_prompt="You are a game theorist and strategist.", temperature=0.1)

        state_evaluation = parsed_content.get("state_evaluation")
        winning_strategy = parsed_content.get("winning_strategy")
        endgame_tactics = parsed_content.get("endgame_tactics")


    for agent in [agent1, agent2]:

        final_prompt = f"""
        Refine the initial game prompt to improve decision-making based on the Game and Strategy.
        ##Initial prompt: {prompt}\n

        **Game:** {game_definition}  
        **Winning Condition:** {winning_condition}  
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

        # Output:
        Provide your answer in the following JSON format:
        ```
        {{
        "optimized_prompt": "string", // The refined prompt that clearly directs decision-making.
        }}
        ``` 
        """
        parsed_content = get_agent_response(agent, final_prompt, system_prompt="You are a game theorist and strategist.", temperature=0.7)

        optimized_prompt = parsed_content.get("optimized_prompt")
        # Current State:
        # The grid is represented as a list of lists, where '1' means the position is still available, and '0' means it is removed:\n{remaining_grid}\n
        # - Winning Strategy: {winning_strategy}\n
        # Current State:\nThe grid is represented as a list of lists, where '1' means the position is still available, and '0' means it is removed:\n{remaining_grid}\n
        one_new_prompt = f"""{optimized_prompt}\n
        
        ### Instructions:
        1. **If a winning move exists, take it immediately.**  
        2. **Otherwise, follow optimal move principles.**  
        3. Justify your move using the extracted strategy.

        # Output:
        Provide your reasoning for the move and the action in the following JSON format:
        ```
        {{
            "reasoning": string,  // Explain why you chose this position (0-based position). 
            "row": integer,       // The row index of the position you select (0-based).
            "col": integer,        // The column index of the position you select (0-based).
        }}
        ```
        """

        retries = 0
        while retries < max_retries:
            parsed_content = get_agent_response(agent, one_new_prompt, system_prompt="You are a skilled Chomp player and debating the best action.")
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
            one_prompt = optimized_prompt
        if i == 1:
            initial_moves['agent2'] = (row, col)
            initial_reasonings['agent2'] = refined_reasoning
            two_prompt = optimized_prompt
        i += 1

    for _ in range(debate_rounds):
        i = 0
        for agent in [agent1, agent2]:
            
            if i == 0:
                # - Winning Strategy: {winning_strategy}\n
                # Current State:\nThe grid is represented as a list of lists, where '1' means the position is still available, and '0' means it is removed:\n{remaining_grid}\n
                prompt = f"""
                {one_prompt}\n 
                
                You initially chose {initial_moves['agent1']} items at first trial by the reason: '{initial_reasonings['agent1']}'.\n
                Other agent argues that you have to choose move as: {initial_moves['agent2']} by the reason: {initial_reasonings['agent2']}.\n
                Considering the other's opinion and your strategy, refine or confirm your move.\n

                ### Instructions:
                1. **If a winning move exists, take it immediately.**  
                2. **Otherwise, follow optimal move principles.**  
                3. Justify your move using the extracted strategy.

                # Output:
                Provide your reasoning for the move and the action in the following JSON format:
                ```
                {{
                    "reasoning": string,  // Explain why you chose this position (0-based position). 
                    "row": integer,       // The row index of the position you select (0-based).
                    "col": integer,        // The column index of the position you select (0-based).
                }}
                ```
                """
            if i == 1:
                prompt = f"""
                {two_prompt}\n

                You initially chose {initial_moves['agent2']} items at first trial by the reason: '{initial_reasonings['agent2']}'.\n
                Other agent argues that you have to choose move as: {initial_moves['agent1']} by the reason: {initial_reasonings['agent1']}.\n
                Considering the other's opinion and your strategy, refine or confirm your move.\n

                ### Instructions:
                1. **If a winning move exists, take it immediately.**  
                2. **Otherwise, follow optimal move principles.**  
                3. Justify your move using the extracted strategy.

                # Output:
                Provide your reasoning for the move and the action in the following JSON format:
                ```
                {{
                    "reasoning": string,  // Explain why you chose this position (0-based position). 
                    "row": integer,       // The row index of the position you select (0-based).
                    "col": integer,        // The column index of the position you select (0-based).
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
                a0_action = (row, col)
                a0_reasoning = refined_reasoning
                
            if i == 1:
                a1_action = (row, col)
                a1_reasoning = refined_reasoning
        
            i += 1

        initial_moves['agent1'] = a0_action
        initial_reasonings['agent1'] = a0_reasoning
        initial_moves['agent2'] = a1_action
        initial_reasonings['agent2'] = a1_reasoning
            
        if len(set(initial_moves.values())) == 1:
            return refined_reasoning, initial_moves['agent1']

    return refined_reasoning, Counter(initial_moves.values()).most_common(1)[0][0]  # Use most common if no consensus


def play_square_chomp_game(size=5, verbose=False):
    with open(f'/home/jihwan/NashIP/result/Chomp_S/{args.agent1_model}_{args.agent1_prompt}_{n_step_lookahead}_{args.agent2_model}_{args.agent2_prompt}.txt', 'a') as f:
        # Initialize the square grid
        grid = [[True for _ in range(size)] for _ in range(size)]
        turn = 0

        while any(any(row) for row in grid):
            current_agent = agents[turn % 2]
            other_agent = agents[(turn + 1) % 2]

            if current_agent["prompting_method"] == "self_consistency":
                reasoning, move = get_consistent_move(current_agent, grid, self_consistency_count)
            elif current_agent["prompting_method"] == "basic":
                reasoning, move = get_basic_move(current_agent, grid)
            elif current_agent["prompting_method"] == "simple":
                reasoning, move = get_move(current_agent, grid)
            elif current_agent["prompting_method"] == "self_reflection":
                reasoning, move = get_move_with_reflection(current_agent, grid)
            elif current_agent["prompting_method"] == "debate":
                reasoning, move = get_move_with_debate(current_agent, current_agent, grid)
            elif current_agent["prompting_method"] == "dreamad":
                reasoning, move = get_move_dreamad(current_agent, current_agent, grid)
            else:
                print("Error: set the prompting methods", file=f)
                return None

            row, col = move

            # Check if the player selected the losing position (N-1, N-1)
            if row == size - 1 and col == size - 1:
                print(f"{current_agent['name']} ate the poisoned square ({row}, {col})! {other_agent['name']} wins!", file=f)
                return other_agent["name"]

            # Apply the move (remove all squares to the left and below)
            apply_move_chomp(grid, row, col)

            if verbose:
                print('Reasoning:', reasoning, '\nAction:', move, file=f)
                print(f"{current_agent['name']} ({current_agent['model']}) takes bite at ({row}, {col}). Remaining state:\n{grid}", file=f)

            if not any(any(row) for row in grid):
                if verbose:
                    print(f"{other_agent['name']} ({other_agent['model']}) wins!", file=f)
                return other_agent["name"]

            turn += 1


def is_valid_move_chomp(grid, row, col):
    size = len(grid)
    return 0 <= row < size and 0 <= col < size and grid[row][col]


def apply_move_chomp(grid, row, col):
    """
    선택한 (row, col) 칸을 포함하여,
    해당 칸보다 **왼쪽 및 아래** 부분을 없앰.
    """
    for r in range(row, -1, -1):  # 아래 방향 (row 이하 전부 제거)
        for c in range(col, -1, -1):  # 왼쪽 방향 (col 이하 전부 제거)
            grid[r][c] = False


def simulate_square_chomp_games(num_games, size=8):
    win_counts = {agent["name"]: 0 for agent in agents}
    with open(f'/home/jihwan/NashIP/result/Chomp_S/{args.agent1_model}_{args.agent1_prompt}_{n_step_lookahead}_{args.agent2_model}_{args.agent2_prompt}.txt', 'a') as f:
        for game_num in range(num_games):
            print(f"\nStarting Game {game_num + 1}", file=f)
            print(f"\nStarting Game {game_num + 1}")
            winner = play_square_chomp_game(size, verbose=True)
            if winner:
                win_counts[winner] += 1

        print("\nGame Results:", file=f)
        for agent in agents:
            win_rate = (win_counts[agent["name"]] / num_games) * 100
            print(f"{agent['name']} Win Rate: {win_rate:.2f}% ({win_counts[agent['name']]} wins out of {num_games})", file=f)

simulate_square_chomp_games(num_games=num_games, size=5)