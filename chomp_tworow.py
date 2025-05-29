import os
import random
from collections import Counter
import argparse
import json
import re
from openai import OpenAI
import time

import google.generativeai as genai
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"),)
parser = argparse.ArgumentParser(description='arguments for training')

parser = argparse.ArgumentParser(description='arguments for training')

parser.add_argument('--agent1_model',     type=str,   default=None, help='model')
parser.add_argument('--agent2_model',     type=str,   default='gpt-4o', help='model')
parser.add_argument('--agent1_prompt',     type=str,   default='basic', help='prompt_method')
parser.add_argument('--agent2_prompt',     type=str,   default='basic', help='prompt_method')
parser.add_argument('--num_games',     type=int,   default=50, help='number of games')

args = parser.parse_args()

num_games = args.num_games  # Number of games to play
num_refine = 3
self_consistency_count = 10  # Number of responses to use for self-consistency
debate_rounds = 3  # Maximum number of debate rounds
max_retries = 5

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
            if agent["model"] in ["gemini-1.5-flash", "gemini-1.5-pro"]:
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
            print('agent model: ', agent["model"])
            print("Error encountered. Retrying in 2 seconds...")
            time.sleep(2)  
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

    available_positions = [
        (r, c) for r, row in enumerate(grid) for c, cell in enumerate(row) if cell
    ]
    if available_positions:
        row, col = random.choice(available_positions)
        reasoning = "Fallback: Randomly selected a position after multiple failed attempts."
        return reasoning, (row, col)

    raise ValueError("No valid moves available, and fallback failed.")

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
    available_positions = [
        (r, c) for r, row in enumerate(grid) for c, cell in enumerate(row) if cell
    ]
    if available_positions:
        row, col = random.choice(available_positions)
        reasoning = "Fallback: Randomly selected a position after multiple failed attempts."
        return reasoning, (row, col)

    raise ValueError("No valid moves available, and fallback failed.")

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

            if is_valid_move_chomp(grid, int(row), int(col)):
                break

            retries += 1

        if retries == max_retries:
            available_positions = [
                (r, c) for r, row_data in enumerate(grid) for c, cell in enumerate(row_data) if cell
            ]
            if available_positions:
                row, col = random.choice(available_positions)
                reasoning = "Fallback: Randomly selected a position after multiple failed attempts."

        action = (int(row), int(col))
        moves.append(action)
        reasoning_list.append(reasoning)

    most_common_move = Counter(moves).most_common(1)[0][0]
    consistent_reasoning = reasoning_list[moves.index(most_common_move)]

    return consistent_reasoning, most_common_move

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

    return refined_reasoning, Counter(initial_moves.values()).most_common(1)[0][0] 

def get_move_with_debate_2(agent1, agent2, judge_agent, grid, debate_rounds=3):
    remaining_grid = [''.join(['1' if cell else '0' for cell in row]) for row in grid]
    prompt_initial = f"""
    # Game Role:
    You are a participant in a game of Chomp.

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
    
    Output your answer as a JSON formatted snippet with keys "reasoning" (string), "row" (integer), and "col" (integer).
    ```
    {{
        "reasoning": string  // Explain why you chose this position.
        "row": integer,  // The row index of the position you select (0-based).
        "col": list     // The column index of the position you select (0-based).
    }}
    ```
    """
    initial_result = get_agent_response(agent1, prompt_initial, system_prompt="You are a skilled Chomp player.", temperature=0.7)
    initial_reasoning = initial_result.get("reasoning")
    row = initial_result.get("row")
    col = initial_result.get("col")
    initial_action = (row, col)
    current_reasoning = initial_reasoning
    current_action = initial_action
    debate_history = f"Game Current State: {prompt_initial}\n Initial Answer: {initial_action} with reasoning: {initial_reasoning}\n"
    for round_num in range(1, debate_rounds + 1):
        prompt_negative = f"{prompt_initial}\n"
        prompt_negative += f"""
        You are {agent2['name']}, playing as the negative side. You disagree with the current answer.
        Your goal is to challenge the current answer and provide reasons why it might be incorrect.
        The current answer is {current_action} with reasoning: '{current_reasoning}'.
        Provide your updated reasoning and an alternative answer.
        
        Output your answer as a JSON formatted snippet with keys "reasoning" (string), "row" (integer), and "col" (integer).
        ```
        {{
            "reasoning": string  // Explain why you chose this position.
            "row": integer,  // The row index of the position you select (0-based).
            "col": list     // The column index of the position you select (0-based).
        }}
    ```
        """
        result_neg = get_agent_response(agent2, prompt_negative, 
                                        system_prompt="You are negative side. Provide your critique and alternative answer as a JSON formatted snippet with keys 'reasoning', 'row', and 'col'.", 
                                        temperature=0.7)
        row = result_neg.get("row")
        col = result_neg.get("col")
        negative_action = (row, col)
        negative_reasoning = result_neg.get("reasoning")
        prompt_affirmative = f"{prompt_initial}\n"
        prompt_affirmative += f"""
        You are {agent1['name']}, playing as the affirmative side. You originally provided the answer {current_action} with reasoning: '{current_reasoning}'.
        After hearing the negative evaluation which suggested an alternative answer of {negative_action} with reasoning: '{negative_reasoning}', please refine or confirm your answer.
        Provide your updated reasoning and answer.
        
        Output your answer as a JSON formatted snippet with keys "reasoning" (string), "row" (integer), and "col" (integer).
        ```
        {{
            "reasoning": string  // Explain why you chose this position.
            "row": integer,  // The row index of the position you select (0-based).
            "col": list     // The column index of the position you select (0-based).
        }}
        ```
        """
        result_affirm = get_agent_response(agent1, prompt_affirmative, 
                                           system_prompt="You are affirmative side. Refine your answer as a JSON formatted snippet with keys 'reasoning', 'row', and 'col' based on the negative feedback.", 
                                           temperature=0.7)
        row = result_affirm.get("row")
        col = result_affirm.get("col")
        refined_action = (row, col)
        refined_reasoning = result_affirm.get("reasoning")
        debate_history += f"Round {round_num}:\n"
        debate_history += f"Negative: action={negative_action}, reasoning='{negative_reasoning}'\n"
        debate_history += f"Affirmative: action={refined_action}, reasoning='{refined_reasoning}'\n"
        current_action = refined_action
        current_reasoning = refined_reasoning
        if negative_action == current_action:
            return current_reasoning, current_action
        
    judge_prompt = f"""
    You are a moderator. There have been two debaters (affirmative and negative) discussing the best move in a game of Chomp variants. 
    Debate Topic: decide which position (row, col) you will select.
    
    Debate History:
    {debate_history}
    
    Based on the debate history, please decide which debater's answer is correct. Output your answer as a JSON formatted snippet with keys "reasoning" (string), "row" (integer), and "col" (integer).
    ```
    {{
        "reasoning": string  // Explain why you chose this position.
        "row": integer,  // The row index of the position you select (0-based).
        "col": list     // The column index of the position you select (0-based).
    }}
    ```
    """
    judge_result = get_agent_response(judge_agent, judge_prompt, 
                                      system_prompt="You are a moderator. Evaluate both sides' arguments and decide which one is correct.", 
                                      temperature=0.7)
    final_reasoning = judge_result.get("reasoning")
    row = judge_result.get("row")
    col = judge_result.get("col")  
    final_action = (row, col)
    
    return final_reasoning, final_action

def get_move_dreamad(agent1, agent2, grid):
    initial_moves = {}
    initial_reasonings = {}
    i = 0
    for agent in [agent1]:
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
        """

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

        ### Format Response as:

        {{
        "optimized_prompt": "string", // The refined prompt that clearly directs decision-making. }}
        """
        parsed_content = get_agent_response(agent, final_prompt, system_prompt="You are a game theorist and strategist.", temperature=0.7)

        optimized_prompt = parsed_content.get("optimized_prompt")
        one_new_prompt = f"""{optimized_prompt}\n

        - Winning Strategy: {winning_strategy}  

        # Current State:
        The grid is represented as binary strings, where '1' means the position is still available, and '0' means it is removed:
        {remaining_grid}
    
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
            "col": integer        // The column index of the position you select (0-based).
        }}
        ```
        """

        retries = 0
        while retries < max_retries:
            parsed_content = get_agent_response(agent, one_new_prompt, system_prompt="You are a skilled Chomp player and debating the best action.")
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
            one_prompt = optimized_prompt
        if i == 1:
            initial_moves['agent2'] = (row, col)
            initial_reasonings['agent2'] = initial_reasoning
            two_prompt = optimized_prompt
        i += 1

    for _ in range(debate_rounds):
        i = 0
        for agent in [agent1, agent2]:
            remaining_grid = [
                ''.join(['1' if cell else '0' for cell in row]) for row in grid
            ]
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
                    "reasoning": string,  // Explain why you chose this position (0-based position). 
                    "row": integer,       // The row index of the position you select (0-based).
                    "col": integer        // The column index of the position you select (0-based).
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

    return refined_reasoning, Counter(initial_moves.values()).most_common(1)[0][0] 


def play_two_row_chomp_game(num_columns=8, verbose=False):
    with open(f'/YourPath/{args.agent1_model}_{args.agent1_prompt}_{args.agent2_model}_{args.agent2_prompt}.txt', 'a') as f:
        grid = [[True for _ in range(num_columns)] for _ in range(2)]
        turn = 0

        while any(any(row) for row in grid):
            current_agent = agents[turn % 2]
            other_agent = agents[(turn + 1) % 2]

            if current_agent["prompting_method"] == "self_consistency":
                reasoning, move = get_consistent_move(current_agent, grid, self_consistency_count)
            elif current_agent["prompting_method"] == "simple":
                reasoning, move = get_move(current_agent, grid)
            elif current_agent["prompting_method"] == "self_reflection":
                reasoning, move = get_move_with_reflection(current_agent, grid)
            elif current_agent["prompting_method"] == "debate":
                reasoning, move = get_move_with_debate(current_agent, current_agent, grid)
            elif current_agent["prompting_method"] == "debate2":
                reasoning, move = get_move_with_debate_2(current_agent, current_agent, current_agent, grid)
            elif current_agent["prompting_method"] == "dreamad":
                reasoning, move = get_move_dreamad(current_agent, current_agent, grid)
            elif current_agent["prompting_method"] == "basic":
                reasoning, move = get_basic_move(current_agent, grid)
            else:
                print("Error: set the prompting methods", file=f)
                return None

            row, col = move
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
    with open(f'/YourPath/{args.agent1_model}_{args.agent1_prompt}_{args.agent2_model}_{args.agent2_prompt}.txt', 'a') as f:
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