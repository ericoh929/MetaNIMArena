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

def get_move(agent, piles):
    remaining_pins = [
        ''.join(['1' if pin else '0' for pin in pile]) for pile in piles
    ]
    prompt = f"""
    # Game Role:
    You are {agent['name']}, a participant in a game of Kayles.

    # Objective:
    Your goal is to win the game by leaving your opponent with no valid moves. The player who takes the last pin(s) wins.

    # Game Rule:
    1. There are two rows of pins (piles).
    2. On your turn, you can remove:
       - A single pin from one pile.
       - Two adjacent pins from one pile.
    3. You cannot remove non-adjacent pins or pins that have already been removed.

    # Current State:
    The rows of pins are represented as binary strings: 
    - '1' means the pin is still available.
    - '0' means the pin has already been removed.
    Current state:
    Pile 1: "{remaining_pins[0]}"
    Pile 2: "{remaining_pins[1]}"

    # Task:
    Based on the current state of the game, decide which pile and pin(s) you will take on this turn.

    # Output:
    Provide the action in the following JSON format:

    ```
    {{
        "pile_index": integer,  // Index of the pile (0 for Pile 1, 1 for Pile 2).
        "pin_indices": list     // A list of integers representing the indices (0-based) of the pins you will remove. Valid moves include single pins or two adjacent pins. Only provide valid indices.
    }}
    ```
    """

    retries = 0
    while retries < max_retries:
        parsed_content = get_agent_response(agent, prompt, system_prompt="You are a skilled Kayles player.")
        reasoning = "None"

        # Ensure the action is valid
        pile_index = parsed_content.get("pile_index")
        pin_indices = parsed_content.get("pin_indices")

        if is_valid_move(piles, pile_index, pin_indices):
            return reasoning, (pile_index, pin_indices)  # Exit loop if the action is valid

        retries += 1

    available_pins = [
        (pile_index, idx) for pile_index, pile in enumerate(piles) for idx, pin in enumerate(pile) if pin
    ]
    if available_pins:
        pile_index, pin_index = random.choice(available_pins)
        pin_indices = [pin_index]
        reasoning = "Fallback: Randomly selected a single available pin after multiple failed attempts."
        return reasoning, (pile_index, pin_indices)

def get_basic_move(agent, piles):
    remaining_pins = [
        ''.join(['1' if pin else '0' for pin in pile]) for pile in piles
    ]
    prompt = f"""
    # Game Role:
    You are {agent['name']}, a participant in a game of Kayles.

    # Objective:
    Your goal is to win the game by leaving your opponent with no valid moves. The player who takes the last pin(s) wins.

    # Game Rule:
    1. There are two rows of pins (piles).
    2. On your turn, you can remove:
       - A single pin from one pile.
       - Two adjacent pins from one pile.
    3. You cannot remove non-adjacent pins or pins that have already been removed.

    # Current State:
    The rows of pins are represented as binary strings: 
    - '1' means the pin is still available.
    - '0' means the pin has already been removed.
    Current state:
    Pile 1: "{remaining_pins[0]}"
    Pile 2: "{remaining_pins[1]}"

    # Task:
    Based on the current state of the game, decide which pile and pin(s) you will take on this turn.

    # Output:
    Provide your reasoning for the move and the action in the following JSON format:

    ```
    {{
        "reasoning": string  // Explain why you chose the pile and the pins to remove.
        "pile_index": integer,  // Index of the pile (0 for Pile 1, 1 for Pile 2).
        "pin_indices": list     // A list of integers representing the indices (0-based) of the pins you will remove. Valid moves include single pins or two adjacent pins. Only provide valid indices.
    }}
    ```
    """

    retries = 0
    while retries < max_retries:
        parsed_content = get_agent_response(agent, prompt, system_prompt="You are a skilled Kayles player.")
        reasoning = parsed_content.get("reasoning")

        # Ensure the action is valid
        pile_index = parsed_content.get("pile_index")
        pin_indices = parsed_content.get("pin_indices")

        if is_valid_move(piles, pile_index, pin_indices):
            return reasoning, (pile_index, pin_indices)  # Exit loop if the action is valid

        retries += 1

    available_pins = [
        (pile_index, idx) for pile_index, pile in enumerate(piles) for idx, pin in enumerate(pile) if pin
    ]
    if available_pins:
        pile_index, pin_index = random.choice(available_pins)
        pin_indices = [pin_index]
        reasoning = "Fallback: Randomly selected a single available pin after multiple failed attempts."
        return reasoning, (pile_index, pin_indices)


# Function for self-consistency: generate multiple responses and choose the most common move
def get_consistent_move(agent, piles, num_responses):
    remaining_pins = [
        ''.join(['1' if pin else '0' for pin in pile]) for pile in piles
    ]
    prompt = f"""
    # Game Role:
    You are {agent['name']}, a participant in a game of Kayles.

    # Objective:
    Your goal is to win the game by leaving your opponent with no valid moves. The player who takes the last pin(s) wins.

    # Game Rule:
    1. There are two rows of pins (piles).
    2. On your turn, you can remove:
       - A single pin from one pile.
       - Two adjacent pins from one pile.
    3. You cannot remove non-adjacent pins or pins that have already been removed.

    # Current State:
    The rows of pins are represented as binary strings: 
    - '1' means the pin is still available.
    - '0' means the pin has already been removed.
    Current state:
    Pile 1: "{remaining_pins[0]}"
    Pile 2: "{remaining_pins[1]}"

    # Task:
    Based on the current state of the game, decide which pile and pin(s) you will take on this turn.

    # Output:
    Provide your reasoning for the move and the action in the following JSON format:

    ```
    {{
        "reasoning": string  // Explain why you chose the pile and the pins to remove.
        "pile_index": integer,  // Index of the pile (0 for Pile 1, 1 for Pile 2).
        "pin_indices": list     // A list of integers representing the indices (0-based) of the pins you will remove. Valid moves include single pins or two adjacent pins. Only provide valid indices.
    }}
    ```
    """

    moves = []
    reasoning_list = []

    for _ in range(num_responses):

        retries = 0
        while retries < max_retries:
            parsed_content = get_agent_response(agent, prompt, system_prompt="You are a skilled Kayles player.")
            reasoning = parsed_content.get("reasoning")

            # Ensure the action is valid
            pile_index = parsed_content.get("pile_index")
            pin_indices = parsed_content.get("pin_indices")

            if is_valid_move(piles, pile_index, pin_indices):
                moves.append((pile_index, pin_indices))
                reasoning_list.append(reasoning)
                break

            retries += 1

        if retries == max_retries:
            available_pins = [
                (pile_index, idx) for pile_index, pile in enumerate(piles) for idx, pin in enumerate(pile) if pin
            ]
            if available_pins:
                pile_index, pin_index = random.choice(available_pins)
                pin_indices = [pin_index]
                reasoning = "Fallback: Randomly selected a single available pin after multiple failed attempts."
                moves.append((pile_index, pin_indices))
                reasoning_list.append(reasoning)

    # Determine the most common valid move
    most_common_move = Counter(moves).most_common(1)[0][0]
    # Get the reasoning for the most common move
    consistent_reasoning = reasoning_list[moves.index(most_common_move)]

    return consistent_reasoning, most_common_move


def get_move_with_reflection(agent, piles):
    remaining_pins = [
        ''.join(['1' if pin else '0' for pin in pile]) for pile in piles
    ]
    prompt_initial = f"""
    # Game Role:
    You are {agent['name']}, a participant in a game of Kayles.

    # Objective:
    Your goal is to win the game by leaving your opponent with no valid moves. The player who takes the last pin(s) wins.

    # Game Rule:
    1. There are two rows of pins (piles).
    2. On your turn, you can remove:
       - A single pin from one pile.
       - Two adjacent pins from one pile.
    3. You cannot remove non-adjacent pins or pins that have already been removed.

    # Current State:
    The rows of pins are represented as binary strings: 
    - '1' means the pin is still available.
    - '0' means the pin has already been removed.
    Current state:
    Pile 1: "{remaining_pins[0]}"
    Pile 2: "{remaining_pins[1]}"

    # Task:
    Based on the current state of the game, decide which pile and pin(s) you will take on this turn.

    # Output:
    Provide your reasoning for the move and the action in the following JSON format:

    ```
    {{
        "reasoning": string  // Explain why you chose the pile and the pins to remove.
        "pile_index": integer,  // Index of the pile (0 for Pile 1, 1 for Pile 2).
        "pin_indices": list     // A list of integers representing the indices (0-based) of the pins you will remove. Valid moves include single pins or two adjacent pins. Only provide valid indices.
    }}
    ```
    """

    retries = 0
    while retries < max_retries:
        parsed_content = get_agent_response(agent, prompt_initial, system_prompt="You are a skilled Kayles player.")
        initial_reasoning = parsed_content.get("reasoning")
        pile_index = parsed_content.get("pile_index")
        pin_indices = parsed_content.get("pin_indices")

        if is_valid_move(piles, pile_index, pin_indices):
            break

        retries += 1

    if retries == max_retries:
        available_pins = [
            (pile_index, idx) for pile_index, pile in enumerate(piles) for idx, pin in enumerate(pile) if pin
        ]
        if available_pins:
            pile_index, pin_index = random.choice(available_pins)
            pin_indices = [pin_index]
            initial_reasoning = "Fallback: Randomly selected a single available pin after multiple failed attempts."

    for k in range(num_refine):

        feedback_prompt = f"""
        # Game Role:
        You are {agent['name']}, a participant in a game of Kayles.

        # Objective:
        Your goal is to win the game by leaving your opponent with no valid moves. The player who takes the last pin(s) wins.

        # Game Rule:
        1. There are two rows of pins (piles).
        2. On your turn, you can remove:
           - A single pin from one pile.
           - Two adjacent pins from one pile.
        3. You cannot remove non-adjacent pins or pins that have already been removed.

        # Current State:
        The rows of pins are represented as binary strings: 
        - '1' means the pin is still available.
        - '0' means the pin has already been removed.
        Current state:
        Pile 1: "{remaining_pins[0]}"
        Pile 2: "{remaining_pins[1]}"

        # Task:
        Based on the current state of the game, give feedback on the first trial's reasoning and action.

        # First trial's reasoning and action:
        You initially chose pile {pile_index} and pins {pin_indices} at first trial by the reason: '{initial_reasoning}'.

        # Output:
        Provide your feedback for the move and the action in the following JSON format:

        ```
        {{
            "feedback": string  // This is the feedback for the initially selected action and reasoning.
        }}
        ```
        """

        parsed_content = get_agent_response(agent, feedback_prompt, system_prompt="You are a skilled Kayles player.")
        feedback = parsed_content.get("feedback")

        refine_prompt = f"""
        # Game Role:
        You are {agent['name']}, a participant in a game of Kayles.

        # Objective:
        Your goal is to win the game by leaving your opponent with no valid moves. The player who takes the last pin(s) wins.

        # Game Rule:
        1. There are two rows of pins (piles).
        2. On your turn, you can remove:
           - A single pin from one pile.
           - Two adjacent pins from one pile.
        3. You cannot remove non-adjacent pins or pins that have already been removed.

        # Current State:
        The rows of pins are represented as binary strings: 
        - '1' means the pin is still available.
        - '0' means the pin has already been removed.
        Current state:
        Pile 1: "{remaining_pins[0]}"
        Pile 2: "{remaining_pins[1]}"

        You initially chose pile {pile_index} and pins {pin_indices} at first trial by the reason: '{initial_reasoning}'.

        You received feedback on your action and reasoning: {feedback}

        # Task:
        Based on the current state of the game and the feedback, refine your reasoning and action. And finally, decide which pile and pin(s) you will take on this turn.

        # Output:
        Provide your reasoning for the move and the action in the following JSON format:

        ```
        {{
            "reasoning": string  // Explain why you chose the pile and the pins to remove.
            "pile_index": integer,  // Index of the pile (0 for Pile 1, 1 for Pile 2).
            "pin_indices": list     // A list of integers representing the indices (0-based) of the pins you will remove. Valid moves include single pins or two adjacent pins. Only provide valid indices.
        }}
        ```
        """

        retries = 0
        while retries < max_retries:
            parsed_content = get_agent_response(agent, refine_prompt, system_prompt="You are a skilled Kayles player.")
            refined_reasoning = parsed_content.get("reasoning")
            pile_index = parsed_content.get("pile_index")
            pin_indices = parsed_content.get("pin_indices")

            if is_valid_move(piles, pile_index, pin_indices):
                break

            retries += 1

        if retries == max_retries:
            available_pins = [
                (pile_index, idx) for pile_index, pile in enumerate(piles) for idx, pin in enumerate(pile) if pin
            ]
            if available_pins:
                pile_index, pin_index = random.choice(available_pins)
                pin_indices = [pin_index]
                refined_reasoning = "Fallback: Randomly selected a single available pin after multiple failed attempts."

    return refined_reasoning, (pile_index, pin_indices)


def bias_removed(agent, pins):
    remaining_pins = ''.join(['1' if pin else '0' for pin in pins])
    first_prompt = f"""
    # Game Role:
    You are {agent['name']}, a participant in a game of Kayles.

    # Objective:
    Your goal is to win the game by leaving your opponent with no valid moves. The player who takes the last pin(s) wins.

    # Game Rule:
    1. There is a single row of pins.
    2. On your turn, you can remove:
       - A single pin.
       - Two adjacent pins.
    3. You cannot remove non-adjacent pins or pins that have already been removed.

    # Current State:
    The row of pins is represented as a binary string: 
    - '1' means the pin is still available.
    - '0' means the pin has already been removed.
    Current state: "{remaining_pins}"

    # Task:
    Based on the current state of the game, decide which pin(s) you will take on this turn.

    # Output:
    Provide your reasoning for the move and the action in the following JSON format:

    ```
    {{
        "reasoning": string  // Explain why you chose the pins to remove.
        "action": list       // A list of integers representing the indices (0-based) of the pins you will remove.
                              // Valid moves include single pins or two adjacent pins. Only provide valid indices.
    }}
    ```
    """

    retries = 0
    while retries < max_retries:
        parsed_content = get_agent_response(agent, first_prompt, system_prompt="You are a skilled Kayles player.")
        refined_reasoning = parsed_content.get("reasoning")
        refined_action = parsed_content.get("action")

        if is_valid_move(pins, refined_action):
            break

        retries += 1

    if retries == max_retries:
        available_pins = [i for i, pin in enumerate(pins) if pin]
        if available_pins:
            refined_action = [random.choice(available_pins)]
            refined_reasoning = "Fallback: Randomly selected a single available pin after multiple failed attempts."

    prompt = f"""Given the following answer, predict the most likely provable question that led to this response.\n
    #Answer:\n
        "reasoning": "{refined_reasoning}",
        "action": {refined_action}\n\n
    The output should be a markdown code snippet formatted in the following schema, including the leading and trailing \\`\\`\\`json" and "\\`\\`\\`":\n\n```\n{{\n\t"provable question": string  // This is a most likely provable question that led to above answer.\n}}
    """

    parsed_content = get_agent_response(agent, prompt, system_prompt="You are a rational smart assistant.")

    question = parsed_content.get("provable question")

    text = f"""Combine the following two instructions into a single instruction that captures their shared intention while harmonizing their nuances. Pay attention to clarity and ensure that any biases in the original instructions are mitigated.

- Original instruction (`{first_prompt}`): The first instruction to consider.
- Bias-mitigated instruction (`{question}`): The second instruction to harmonize.

    The output should be a markdown code snippet formatted in the following schema, including the leading and trailing \\`\\`\\`json" and "\\`\\`\\`":\n\n```\n{{\n\t"instruction": string  // This is the combined instruction harmonizing the two instructions.\n\t"reasoning": string  // This is the reason why new instruction is harmonized.}}
    """

    parsed_content = get_agent_response(agent, text, system_prompt="You are a rational smart assistant.")

    new_instruction = parsed_content.get("instruction")

    text = f"""{new_instruction}

    # Output:
    Provide your reasoning for the move and the action in the following JSON format:

    ```
    {{
        "reasoning": string  // Explain why you chose the pins to remove.
        "action": list       // A list of integers representing the indices (0-based) of the pins you will remove.
                              // Valid moves include single pins or two adjacent pins. Only provide valid indices.
    }}
    ```
    """
    retries = 0
    while retries < max_retries:
        parsed_content = get_agent_response(agent, text, system_prompt="You are a rational player.")
        refined_reasoning = parsed_content.get("reasoning")
        refined_action = parsed_content.get("action")

        if is_valid_move(pins, refined_action):
            break

        retries += 1

    if retries == max_retries:
        available_pins = [i for i, pin in enumerate(pins) if pin]
        if available_pins:
            refined_action = [random.choice(available_pins)]
            refined_reasoning = "Fallback: Randomly selected a single available pin after multiple failed attempts."
    
    return refined_reasoning, refined_action

def bias_mitigated(agent, piles):
    remaining_pins = [
        ''.join(['1' if pin else '0' for pin in pile]) for pile in piles
    ]
    first_prompt = f"""
    # Game Role:
    You are {agent['name']}, a participant in a game of Kayles.

    # Objective:
    Your goal is to win the game by leaving your opponent with no valid moves. The player who takes the last pin(s) wins.

    # Game Rule:
    1. There are two rows of pins (piles).
    2. On your turn, you can remove:
       - A single pin from one pile.
       - Two adjacent pins from one pile.
    3. You cannot remove non-adjacent pins or pins that have already been removed.

    # Current State:
    The rows of pins are represented as binary strings: 
    - '1' means the pin is still available.
    - '0' means the pin has already been removed.
    Current state:
    Pile 1: "{remaining_pins[0]}"
    Pile 2: "{remaining_pins[1]}"

    # Task:
    Based on the current state of the game, decide which pile and pin(s) you will take on this turn.

    # Output:
    Provide your reasoning for the move and the action in the following JSON format:

    ```
    {{
        "reasoning": string  // Explain why you chose the pile and the pins to remove.
        "pile_index": integer,  // Index of the pile (0 for Pile 1, 1 for Pile 2).
        "pin_indices": list     // A list of integers representing the indices (0-based) of the pins you will remove. Valid moves include single pins or two adjacent pins. Only provide valid indices.
    }}
    ```
    """

    prompt = f"""Given the following instruction, rewrite it to minimize bias stemming from strong prior knowledge while preserving its original intent and clarity.\n
    #Instruction:{first_prompt}\n

    The output should be a markdown code snippet formatted in the following schema, including the leading and trailing \`\`\`json" and "\`\`\`":\n\n```
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

    The output should be a markdown code snippet formatted in the following schema, including the leading and trailing \`\`\`json" and "\`\`\`":\n\n```
    {{
        "instruction": string  // This is the combined instruction harmonizing the two instructions.
        "reasoning": string  // This is the reason why new instruction is harmonized.
    }}
    ```
    """

    parsed_content = get_agent_response(agent, text, system_prompt="You are a rational smart assistant.")

    new_instruction = parsed_content.get("instruction")

    new_instruction = f"""{new_instruction}\n\n
    # Output:
    Provide your reasoning for the move and the action in the following JSON format:

    ```
    {{
        "reasoning": string  // Explain why you chose the pile and the pins to remove.
        "pile_index": integer,  // Index of the pile (0 for Pile 1, 1 for Pile 2).
        "pin_indices": list     // A list of integers representing the indices (0-based) of the pins you will remove. Valid moves include single pins or two adjacent pins. Only provide valid indices.
    }}
    ```
    """

    retries = 0
    while retries < max_retries:
        parsed_content = get_agent_response(agent, new_instruction, system_prompt="You are a rational game player.")
        refined_reasoning = parsed_content.get("reasoning")
        pile_index = parsed_content.get("pile_index")
        pin_indices = parsed_content.get("pin_indices")

        if is_valid_move(piles, pile_index, pin_indices):
            break

        retries += 1

    if retries == max_retries:
        available_pins = [
            (pile_index, idx) for pile_index, pile in enumerate(piles) for idx, pin in enumerate(pile) if pin
        ]
        if available_pins:
            pile_index, pin_index = random.choice(available_pins)
            pin_indices = [pin_index]
            refined_reasoning = "Fallback: Randomly selected a single available pin after multiple failed attempts."

    return refined_reasoning, (pile_index, pin_indices)

def get_move_with_debate(agent1, agent2, piles):
    initial_moves = {}
    initial_reasonings = {}
    i = 0
    for agent in [agent1, agent2]:
        remaining_pins = [
            ''.join(['1' if pin else '0' for pin in pile]) for pile in piles
        ]
        prompt = f"""
        # Game Role:
        You are {agent['name']}, a participant in a game of Kayles.

        # Objective:
        Your goal is to win the game by leaving your opponent with no valid moves. The player who takes the last pin(s) wins.

        # Game Rule:
        1. There are two rows of pins (piles).
        2. On your turn, you can remove:
           - A single pin from one pile.
           - Two adjacent pins from one pile.
        3. You cannot remove non-adjacent pins or pins that have already been removed.

        # Current State:
        The rows of pins are represented as binary strings: 
        - '1' means the pin is still available.
        - '0' means the pin has already been removed.
        Current state:
        Pile 1: "{remaining_pins[0]}"
        Pile 2: "{remaining_pins[1]}"

        # Task:
        Based on the current state of the game, decide which pile and pin(s) you will take on this turn.

        # Output:
        Provide your reasoning for the move and the action in the following JSON format:

        ```
        {{
            "reasoning": string  // Explain why you chose the pile and the pins to remove.
            "pile_index": integer,  // Index of the pile (0 for Pile 1, 1 for Pile 2).
            "pin_indices": list     // A list of integers representing the indices (0-based) of the pins you will remove. Valid moves include single pins or two adjacent pins. Only provide valid indices.
        }}
        ```
        """

        retries = 0
        while retries < max_retries:
            parsed_content = get_agent_response(agent, prompt, system_prompt="You are a skilled Kayles player and debating the best action.")
            initial_reasoning = parsed_content.get("reasoning")
            pile_index = parsed_content.get("pile_index")
            pin_indices = parsed_content.get("pin_indices")

            if is_valid_move(piles, pile_index, pin_indices):
                break

            retries += 1

        if retries == max_retries:
            available_pins = [
                (pile_index, idx) for pile_index, pile in enumerate(piles) for idx, pin in enumerate(pile) if pin
            ]
            if available_pins:
                pile_index, pin_index = random.choice(available_pins)
                pin_indices = [pin_index]
                initial_reasoning = "Fallback: Randomly selected a single available pin after multiple failed attempts."

        if i == 0:
            initial_moves['agent1'] = (pile_index, pin_indices)
            initial_reasonings['agent1'] = initial_reasoning
        if i == 1:
            initial_moves['agent2'] = (pile_index, pin_indices)
            initial_reasonings['agent2'] = initial_reasoning
        i += 1

    for _ in range(debate_rounds):
        i = 0
        for agent in [agent1, agent2]:
            if i == 0:
                remaining_pins = [
                    ''.join(['1' if pin else '0' for pin in pile]) for pile in piles
                ]
                prompt = f"""
                # Game Role:
                You are {agent['name']}, a participant in a game of Kayles.

                # Objective:
                Your goal is to win the game by leaving your opponent with no valid moves. The player who takes the last pin(s) wins.

                # Game Rule:
                1. There are two rows of pins (piles).
                2. On your turn, you can remove:
                   - A single pin from one pile.
                   - Two adjacent pins from one pile.
                3. You cannot remove non-adjacent pins or pins that have already been removed.

                # Current State:
                The rows of pins are represented as binary strings: 
                - '1' means the pin is still available.
                - '0' means the pin has already been removed.
                Current state:
                Pile 1: "{remaining_pins[0]}"
                Pile 2: "{remaining_pins[1]}"

                # Task:
                Based on the current state of the game, decide which pile and pin(s) you will take on this turn.

                You initially chose {initial_moves['agent1'][1]} from Pile {initial_moves['agent1'][0] + 1} at first trial by the reason: '{initial_reasonings['agent1']}'.
                Other agent argues that you have to choose move as: {initial_moves['agent2'][1]} from Pile {initial_moves['agent2'][0] + 1} by the reason: {initial_reasonings['agent2']}.
                Considering the other's opinion, refine or confirm your move.

                # Output:
                Provide your reasoning for the move and the action in the following JSON format:

                ```
                {{
                    "reasoning": string  // Explain why you chose the pile and the pins to remove.
                    "pile_index": integer,  // Index of the pile (0 for Pile 1, 1 for Pile 2).
                    "pin_indices": list     // A list of integers representing the indices (0-based) of the pins you will remove. Valid moves include single pins or two adjacent pins. Only provide valid indices.
                }}
                ```
                """

            if i == 1:
                remaining_pins = [
                    ''.join(['1' if pin else '0' for pin in pile]) for pile in piles
                ]
                prompt = f"""
                # Game Role:
                You are {agent['name']}, a participant in a game of Kayles.

                # Objective:
                Your goal is to win the game by leaving your opponent with no valid moves. The player who takes the last pin(s) wins.

                # Game Rule:
                1. There are two rows of pins (piles).
                2. On your turn, you can remove:
                   - A single pin from one pile.
                   - Two adjacent pins from one pile.
                3. You cannot remove non-adjacent pins or pins that have already been removed.

                # Current State:
                The rows of pins are represented as binary strings: 
                - '1' means the pin is still available.
                - '0' means the pin has already been removed.
                Current state:
                Pile 1: "{remaining_pins[0]}"
                Pile 2: "{remaining_pins[1]}"

                # Task:
                Based on the current state of the game, decide which pile and pin(s) you will take on this turn.

                You initially chose {initial_moves['agent2'][1]} from Pile {initial_moves['agent2'][0] + 1} at first trial by the reason: '{initial_reasonings['agent2']}'.
                Other agent argues that you have to choose move as: {initial_moves['agent1'][1]} from Pile {initial_moves['agent1'][0] + 1} by the reason: {initial_reasonings['agent1']}.
                Considering the other's opinion, refine or confirm your move.

                # Output:
                Provide your reasoning for the move and the action in the following JSON format:

                ```
                {{
                    "reasoning": string  // Explain why you chose the pile and the pins to remove.
                    "pile_index": integer,  // Index of the pile (0 for Pile 1, 1 for Pile 2).
                    "pin_indices": list     // A list of integers representing the indices (0-based) of the pins you will remove. Valid moves include single pins or two adjacent pins. Only provide valid indices.
                }}
                ```
                """

            retries = 0
            while retries < max_retries:
                parsed_content = get_agent_response(agent, prompt, system_prompt="You are a skilled Kayles player and debating the best action.")
                refined_reasoning = parsed_content.get("reasoning")
                pile_index = parsed_content.get("pile_index")
                pin_indices = parsed_content.get("pin_indices")

                if is_valid_move(piles, pile_index, pin_indices):
                    break

                retries += 1

            if retries == max_retries:
                available_pins = [
                    (pile_index, idx) for pile_index, pile in enumerate(piles) for idx, pin in enumerate(pile) if pin
                ]
                if available_pins:
                    pile_index, pin_index = random.choice(available_pins)
                    pin_indices = [pin_index]
                    refined_reasoning = "Fallback: Randomly selected a single available pin after multiple failed attempts."

            if i == 0:
                a0_action = (pile_index, tuple(pin_indices))
                a0_reasoning = initial_reasoning
                
                # print('debate round:, ', t, 'my action: ', initial_action)
                # print('debate round:, ', t, 'my reasoning: ', initial_reasoning)    
            if i == 1:
                a1_action = (pile_index, tuple(pin_indices))
                a1_reasoning = initial_reasoning
                
                # print('debate round:, ', t, 'others action: ', initial_action)
                # print('debate round:, ', t, 'others reasoning: ', initial_reasoning)   
            i += 1
        initial_moves['agent1'] = a0_action
        initial_reasonings['agent1'] = a0_reasoning
        initial_moves['agent2'] = a1_action
        initial_reasonings['agent2'] = a1_reasoning

        print('initial moves:: ', initial_moves)
        normalized_moves = {agent: (pile_index, tuple(pin_indices)) for agent, (pile_index, pin_indices) in initial_moves.items()}
        if len(set(normalized_moves.values())) == 1:
            return refined_reasoning, initial_moves['agent1']

    return refined_reasoning, Counter(normalized_moves.values()).most_common(1)[0][0]  # Use most common if no consensus

def get_move_dreamad(agent1, agent2, piles):
    initial_moves = {}
    initial_reasonings = {}
    i = 0
    for agent in [agent1]:
        remaining_pins = [
            ''.join(['1' if pin else '0' for pin in pile]) for pile in piles
        ]
        prompt = f"""
        # Game Role:
        You are {agent['name']}, a participant in a game of Kayles.

        # Objective:
        Your goal is to win the game by leaving your opponent with no valid moves. The player who takes the last pin(s) wins.

        # Game Rule:
        1. There are two rows of pins (piles).
        2. On your turn, you can remove:
           - A single pin from one pile.
           - Two adjacent pins from one pile.
        3. You cannot remove non-adjacent pins or pins that have already been removed.

        # Current State:
        The rows of pins are represented as binary strings: 
        - '1' means the pin is still available.
        - '0' means the pin has already been removed.
        Current state:
        Pile 1: "{remaining_pins[0]}"
        Pile 2: "{remaining_pins[1]}"

        # Task:
        Based on the current state of the game, decide which pile and pin(s) you will take on this turn.
        """
        #################prompt1#################
        game_prompt = f"""
        Below is a game description. Extract key information.

        **Game Description:**
        {prompt}

        ### Format Response as:
        {{
        "game_definition": "string", // What is the definition of this game?.
        "winning_condition": "string", // How to win the game.
        "move_constraints": "string" // What actions are allowed per turn.
        }}
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

        ### Format Response as:
        {{
        "state_evaluation": "string", // How to assess the game state.
        "winning_strategy": "string", // Winning strategy in this turn to win this game.
        "endgame_tactics": "string" // Best strategy in a near-win situation.}}
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
        ```
        {{
        "optimized_prompt": "string", // The refined prompt that clearly directs decision-making. 
        }}
        ```
        """
        parsed_content = get_agent_response(agent, final_prompt, system_prompt="You are a game theorist and strategist.", temperature=0.7)

        optimized_prompt = parsed_content.get("optimized_prompt")

        one_new_prompt = f"""{optimized_prompt}\n

        # Current State:
        The rows of pins are represented as binary strings: 
        - '1' means the pin is still available.
        - '0' means the pin has already been removed.
        Current state:
        Pile 1: "{remaining_pins[0]}"
        Pile 2: "{remaining_pins[1]}"
    
        ### Instructions:
        1. **If a winning move exists, take it immediately.**  
        2. **Otherwise, follow optimal move principles.**  
        3. Justify your move using the extracted strategy.

        # Output: Provide your reasoning for the move and the action in the following JSON format:
        ```
        {{
            "reasoning": string  // Explain why you chose the pile and the pins to remove.
            "pile_index": integer,  // Index of the pile (0 for Pile 1, 1 for Pile 2). Do not print None type.
            "pin_indices": list     // A list of integers representing the indices (0-based) of the pins you will remove. Valid moves include single pins or two adjacent pins. Only provide valid indices. Do not print None type.
        }}
        ```
        """

        retries = 0
        while retries < max_retries:
            one_parsed_content = get_agent_response(agent, one_new_prompt, system_prompt="You are a game theorist and strategist.")
            refined_reasoning = one_parsed_content.get("reasoning")
            pile_index = one_parsed_content.get("pile_index")
            pin_indices = one_parsed_content.get("pin_indices")

            if is_valid_move(piles, pile_index, pin_indices):
                break

            retries += 1

        if retries == max_retries:
            available_pins = [
                (pile_index, idx) for pile_index, pile in enumerate(piles) for idx, pin in enumerate(pile) if pin
            ]
            if available_pins:
                pile_index, pin_index = random.choice(available_pins)
                pin_indices = [pin_index]
                refined_reasoning = "Fallback: Randomly selected a single available pin after multiple failed attempts."

        if i == 0:
            initial_moves['agent1'] = (pile_index, tuple(pin_indices))
            initial_reasonings['agent1'] = refined_reasoning
            one_prompt = optimized_prompt
        if i == 1:
            initial_moves['agent2'] = (pile_index, tuple(pin_indices))
            initial_reasonings['agent2'] = refined_reasoning
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

                ### Instructions:
                1. **If a winning move exists, take it immediately.**  
                2. **Otherwise, follow optimal move principles.**  
                3. Justify your move using the extracted strategy.

                # Output: 
                Provide your reasoning for the move and the action in the following JSON format:
                ```
                {{
                    "reasoning": string  // Explain why you chose the pile and the pins to remove.
                    "pile_index": integer,  // Index of the pile (0 for Pile 1, 1 for Pile 2). Do not print None type.
                    "pin_indices": list     // A list of integers representing the indices (0-based) of the pins you will remove. Valid moves include single pins or two adjacent pins. Only provide valid indices. Do not print None type.
                }}
                ```
                """
            if i == 1:
                # Current State:
                # The rows of pins are represented as binary strings: 
                # - '1' means the pin is still available.
                # - '0' means the pin has already been removed.
                # Current state:
                # Pile 1: "{remaining_pins[0]}"
                # Pile 2: "{remaining_pins[1]}"
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
                    "reasoning": string  // Explain why you chose the pile and the pins to remove.
                    "pile_index": integer,  // Index of the pile (0 for Pile 1, 1 for Pile 2). Do not print None type.
                    "pin_indices": list     // A list of integers representing the indices (0-based) of the pins you will remove. Valid moves include single pins or two adjacent pins. Only provide valid indices. Do not print None type.
                }}
                ```
                """
            retries = 0
            while retries < max_retries:
                one_parsed_content = get_agent_response(agent, prompt, system_prompt="You are a game theorist and strategist.")
                # print('output:', one_parsed_content)
                refined_reasoning = one_parsed_content.get("reasoning")
                pile_index = one_parsed_content.get("pile_index")
                pin_indices = one_parsed_content.get("pin_indices")
                # print('pile index: ', pile_index)
                # print('pin indices: ', pin_indices)

                if is_valid_move(piles, pile_index, pin_indices):
                    break

                retries += 1

            if retries == max_retries:
                available_pins = [
                    (pile_index, idx) for pile_index, pile in enumerate(piles) for idx, pin in enumerate(pile) if pin
                ]
                if available_pins:
                    pile_index, pin_index = random.choice(available_pins)
                    pin_indices = [pin_index]
                    refined_reasoning = "Fallback: Randomly selected a single available pin after multiple failed attempts."

            if i == 0:
                a0_action = (pile_index, tuple(pin_indices))
                a0_reasoning = refined_reasoning
                
            if i == 1:
                a1_action = (pile_index, tuple(pin_indices))
                a1_reasoning = refined_reasoning
        
            i += 1
        initial_moves['agent1'] = a0_action
        initial_reasonings['agent1'] = a0_reasoning
        initial_moves['agent2'] = a1_action
        initial_reasonings['agent2'] = a1_reasoning

        normalized_moves = {agent: (pile_index, tuple(pin_indices)) for agent, (pile_index, pin_indices) in 
        initial_moves.items()}
        # print('normal moves: ', normalized_moves)
        if len(set(normalized_moves.values())) == 1:
            return refined_reasoning, initial_moves['agent1']

    return refined_reasoning, Counter(normalized_moves.values()).most_common(1)[0][0]  # Use most common if no consensus


import random

def play_kayles_game_two_piles(total_pins_pile1, total_pins_pile2, verbose=False):
    with open(f'/home/jihwan/NashIP/result/2K/{args.agent1_model}_{args.agent1_prompt}_{n_step_lookahead}_{args.agent2_model}_{args.agent2_prompt}.txt', 'a') as f:
        pile1 = [True] * total_pins_pile1
        pile2 = [True] * total_pins_pile2
        turn = 0

        while any(pile1) or any(pile2):
            current_agent = agents[turn % 2]
            other_agent = agents[(turn + 1) % 2]

            if current_agent["prompting_method"] == "self_consistency":
                reasoning, move = get_consistent_move(current_agent, [pile1, pile2], self_consistency_count)
            elif current_agent["prompting_method"] == "self_reflection":
                reasoning, move = get_move_with_reflection(current_agent, [pile1, pile2])
            elif current_agent["prompting_method"] == "debate":
                reasoning, move = get_move_with_debate(current_agent, current_agent, [pile1, pile2])
            elif current_agent["prompting_method"] == "dreamad":
                reasoning, move = get_move_dreamad(current_agent, current_agent, [pile1, pile2])
            elif current_agent["prompting_method"] == "bias_removed":
                reasoning, move = bias_removed(current_agent, [pile1, pile2])
            elif current_agent["prompting_method"] == "bias_mitigated":
                reasoning, move = bias_mitigated(current_agent, [pile1, pile2])
            elif current_agent["prompting_method"] == "basic":
                reasoning, move = get_basic_move(current_agent, [pile1, pile2])
            elif current_agent["prompting_method"] == "simple":
                reasoning, move = get_move(current_agent, [pile1, pile2])
            else:
                print("Error: set the prompting methods")
                return None
            
            print('move: ', move)

            pile_index, pin_indices = move

            # ✅ 유효하지 않은 이동이면 랜덤한 유효한 액션 수행
            if not is_valid_move([pile1, pile2], pile_index, pin_indices):
                print(f"Invalid move by {current_agent['name']}. Choosing random valid move.", file=f)
                move = get_random_valid_move_two_piles([pile1, pile2])  # 랜덤 액션 수행
                pile_index, pin_indices = move

            # ✅ 랜덤 선택한 액션 적용
            apply_move([pile1, pile2], pile_index, pin_indices)

            if verbose:
                print('Reasoning:', reasoning, '\nAction:', move, file=f)
                print(f"{current_agent['name']} ({current_agent['model']}) removes pins {pin_indices} from pile {pile_index + 1}. Remaining state: {[pile1, pile2]}", file=f)

            if not any(pile1) and not any(pile2):
                if verbose:
                    print(f"{current_agent['name']} ({current_agent['model']}) wins!", file=f)
                return current_agent["name"]

            turn += 1


def is_valid_move(piles, pile_index, move):
    """ 선택한 핀이 유효한지 확인 """
    # print('piles, pile index, move: ', piles, pile_index, move)
    pile = piles[pile_index]
    if len(move) == 1:  # 단일 핀 제거
        return 0 <= move[0] < len(pile) and pile[move[0]]
    elif len(move) == 2:  # 인접한 두 핀 제거
        return (
            0 <= move[0] < len(pile)
            and 0 <= move[1] < len(pile)
            and abs(move[0] - move[1]) == 1
            and pile[move[0]]
            and pile[move[1]]
        )
    return False


def apply_move(piles, pile_index, move):
    """ 선택한 핀 제거 """
    pile = piles[pile_index]
    for index in move:
        pile[index] = False


def get_random_valid_move_two_piles(piles):
    """ 랜덤한 유효한 액션 선택 (가능하면 두 개의 인접한 핀 제거, 그렇지 않으면 하나만 제거) """
    available_pins_pile1 = [i for i, pin in enumerate(piles[0]) if pin]
    available_pins_pile2 = [i for i, pin in enumerate(piles[1]) if pin]

    if not available_pins_pile1 and not available_pins_pile2:
        return None  # 모든 핀이 제거됨

    # ✅ 무작위로 하나의 더미(pile) 선택
    chosen_pile = 0 if (available_pins_pile1 and (not available_pins_pile2 or random.random() < 0.5)) else 1
    available_pins = available_pins_pile1 if chosen_pile == 0 else available_pins_pile2

    # ✅ 인접한 두 개의 핀을 선택할 확률 50%
    if len(available_pins) > 1:
        random.shuffle(available_pins)
        for i in range(len(available_pins) - 1):
            if available_pins[i] + 1 == available_pins[i + 1]:  # 인접한 핀 찾기
                return (chosen_pile, (available_pins[i], available_pins[i + 1]))

    # ✅ 그렇지 않으면 단일 핀 제거
    return (chosen_pile, (random.choice(available_pins),))


# Run the simulation
def simulate_kayles_games_two_piles(num_games, total_pins_pile1, total_pins_pile2):
    win_counts = {agent["name"]: 0 for agent in agents}
    with open(f'/home/jihwan/NashIP/result/2K/{args.agent1_model}_{args.agent1_prompt}_{n_step_lookahead}_{args.agent2_model}_{args.agent2_prompt}.txt', 'a') as f:
        for game_num in range(num_games):
            print(f"\nStarting Game {game_num + 1}", file=f)
            print(f"\nStarting Game {game_num + 1}")
            winner = play_kayles_game_two_piles(total_pins_pile1, total_pins_pile2, verbose=True)
            if winner:
                win_counts[winner] += 1

        print("\nGame Results:", file=f)
        for agent in agents:
            win_rate = (win_counts[agent["name"]] / num_games) * 100
            print(f"{agent['name']} Win Rate: {win_rate:.2f}% ({win_counts[agent['name']]} wins out of {num_games})", file=f)


simulate_kayles_games_two_piles(num_games, total_pins_pile1=5, total_pins_pile2=6)