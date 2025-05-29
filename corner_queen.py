import os
import random
from collections import Counter
import argparse
import json
import re
from openai import OpenAI
import time

import google.generativeai as genai

genai.configure(api_key='Your GEMINI KEY')
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"),)


parser = argparse.ArgumentParser(description='arguments for training')

parser.add_argument('--agent1_model',     type=str,   default=None,      help='model')
parser.add_argument('--agent2_model',     type=str,   default='gpt-4o',  help='model')
parser.add_argument('--agent1_prompt',    type=str,   default='basic',   help='prompt_method')
parser.add_argument('--agent2_prompt',    type=str,   default='basic',   help='prompt_method')
parser.add_argument('--num_games',        type=int,   default='50',      help='number of games')

args = parser.parse_args()

board_height = 16
board_width = 20
initial_queen_position = [4, 16] 

num_games = args.num_games
num_refine = 3
self_consistency_count = 10
debate_rounds = 3
max_retries = 5

def get_rule_and_state_prompt(board_height, board_width, position):
    r, c = tuple(position)
    return f"""

    # Game Rule:
    - Board dimensions: {board_height}(Height) x {board_width}(Width)
    - Coordinates are given as [row, col] with indices starting from 0.
        - row=0 is the top row, and row increases as you go downward.
        - col=0 is the leftmost column, and col increases as you move to the right.
        - therefore, valid range for row is 0 to{board_height-1}, and for col is 0 to {board_width-1}.
    
    - From the current position [r, c], players should take the move which is one of the followings:
        1) Move Left:       [r, c'] where c' < c and r stays the same, or
        2) Move Down:       [r', c] where r' > r and c stays the same, or
        3) Move towards Left-down diagonal: [r + d, c - d] for some integer d > 0.

    - The game ends when the queen reaches the left-down corner [row={board_height-1}, col=0].
    
    # Current State:
    - Current position: [row={r}, col={c}]
    """

def get_game_rule_prompt(agent_name, board_height, board_width, position):
    
    return f"""
            # Game Role:
            You are {agent_name}, a participant in a Corner Queen game. 

            # Objective:
            Move the queen so that you are the first to place the leftdown corner.
            """ + get_rule_and_state_prompt(board_height, board_width, position)

def get_standard_decision_instruction():
    return """
    # Task:
    Based on the current state of the game, decide which square to move to [row, col].

    # Output:
    Provide your reasoning for the move and the action in the following JSON format:
    ```
    {
        "reasoning":  string  // Explain why you chose the move.
        "action": [row, col] // The square you will move to [row, col]
    }
    ```
    """

def is_in_corner(position, board_width, board_height):
    corners = [
        [board_height - 1, 0]
    ]
    return position in corners

def is_valid_move(current_pos, next_pos, board_width, board_height):

    r1, c1 = tuple(current_pos)
    r2, c2 = tuple(next_pos)

    if not (isinstance(next_pos, list) and len(next_pos) == 2):
        return False

    if not (0 <= r2 < board_height and 0 <= c2 < board_width):
        return False
    
    if r1 == r2 and c1 == c2:
        return False

    if r1 == r2 and c2 < c1:
        return True

    if c1 == c2 and r2 > r1:
        return True

    row_diff = r2 - r1
    col_diff = c1 - c2   
    if row_diff > 0 and col_diff > 0 and (row_diff == col_diff):
        return True

    return False

def apply_move(position, next_pos):
    position[0] = next_pos[0]
    position[1] = next_pos[1]

def get_random_valid_move(position, board_width, board_height):
    valid_candidates = []
    for nr in range(board_height):
        for nc in range(board_width):
            if is_valid_move(position, [nr, nc], board_width, board_height):
                valid_candidates.append([nr, nc])
    if not valid_candidates:
        return None
    return random.choice(valid_candidates)


def get_valid_move_with_retry(agent, prompt, system_prompt, position, board_width, board_height, fallback_reason="Fallback: random valid move"):
    for _ in range(max_retries):
        parsed_content = get_agent_response(agent, prompt, system_prompt)
        print('parsed content: ', parsed_content)
        if not parsed_content:
            continue

        action = parsed_content.get("action", [])
        reasoning = parsed_content.get("reasoning", "No reasoning provided.")

        if len(action) == 2 and is_valid_move(position, action, board_width, board_height):
            return reasoning, action

    fallback = get_random_valid_move(position, board_width, board_height)
    if fallback:
        return fallback_reason, fallback
    return "No valid moves", []

def get_agent_response(agent, prompt, system_prompt="You are a skilled Corner Queen player. Output your answer as a valid JSON with keys 'reasoning' and 'action'.", temperature=0.7):
    def parse_content(response_str, split_key):
        try:
            if split_key not in response_str:
                return None
            content = response_str.split(split_key)[1]
            match = re.search(r'\{.*?\}', content, re.DOTALL)
            parsed_content = match.group(0).replace('\xa0', '').strip() if match else None
            return json.loads(parsed_content) if parsed_content else None
        except (AttributeError, json.JSONDecodeError, IndexError):
            return None

    while True:
        try:
            if agent["model"] in ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo", "gpt-o3-mini"]:
                response = client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    model=agent["model"],
                    temperature=temperature,
                )
                content = response.choices[0].message.content
                match = re.search(r'\{.*?\}', content, re.DOTALL)
                if match:
                    json_str = match.group(0).replace('\xa0', '').strip()
                    parsed = json.loads(json_str)
                    if parsed:
                        return parsed
                    
            elif agent["model"] in ["gemini-1.5-flash", "gemini-1.5-pro"]:
                generation_config = {
                    "temperature": temperature,
                    "top_p": 0.95,
                    "top_k": 40,
                    "max_output_tokens": 8192,
                    "response_mime_type": "application/json",
                }
                gen_model = genai.GenerativeModel(
                    model_name=agent["model"],
                    generation_config=generation_config,
                    system_instruction=system_prompt,
                )
                chat_session = gen_model.start_chat(history=[])
                response = chat_session.send_message(prompt)
                try:
                    return json.loads(response.text)
                except json.JSONDecodeError:
                    print("Error decoding JSON for gemini model. Retrying...")

            else:
                split_key = "<|start_header_id|>assistant<|end_header_id|>\n"
                response_str = ""  
                parsed_content = parse_content(response_str, split_key)
                if parsed_content:
                    return parsed_content

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


def get_move(agent, position, board_width, board_height):
    prompt = get_game_rule_prompt(agent["name"], board_height, board_width, position) + '\n' + """

    # Task:
    Based on the current state of the game, decide which square to move to [row, col].

    # Output:
    Provide your action in the following JSON format:
    ```
    {
        "action": [row, col] // The square you will move to
    }
    ```
    """
    reasoning, action = get_valid_move_with_retry(agent, prompt, "You are a skilled Corner Queen player. Output your answer as a valid JSON with keys 'action'.", position, board_width, board_height)
    print(action)
    return reasoning, action

def get_basic_move(agent, position, board_width, board_height):
    prompt = get_game_rule_prompt(agent["name"], board_height, board_width, position) + '\n' + get_standard_decision_instruction()
    reasoning, action = get_valid_move_with_retry(agent, prompt, "You are a skilled Corner Queen player.", position, board_width, board_height)
    return reasoning, action


def get_consistent_move(agent, position, board_width, board_height, num_responses=5):
    prompt = get_game_rule_prompt(agent["name"], board_height, board_width, position) + '\n' + get_standard_decision_instruction()

    moves = []
    reasoning_list = []

    for _ in range(num_responses):
        reasoning, action = get_valid_move_with_retry(
            agent, prompt, "You are a skilled Corner Queen player.",
            position, board_width, board_height
        )
        moves.append(action)
        reasoning_list.append(reasoning)

    if not moves:
        return "No valid moves were found in self-consistency sampling.", []

    action_counts = Counter(tuple(move) for move in moves)
    most_common_move = list(action_counts.most_common(1)[0][0])

    return "Self-consistency chosen move", most_common_move

def get_move_with_reflection(agent, position, board_width, board_height):
    prompt_initial = get_game_rule_prompt(agent["name"], board_height, board_width, position) \
        + '\n' + get_standard_decision_instruction()
    
    initial_reasoning, initial_move = get_valid_move_with_retry(
        agent, prompt_initial, "You are a skilled Corner Queen player.", position, board_width, board_height,
        fallback_reason="Fallback: Randomly selected a valid queen move after multiple attempts."
    )

    if not initial_move:
        return initial_reasoning, []

    refined_action = initial_move
    refined_reasoning = initial_reasoning

    for _ in range(num_refine):
        feedback_prompt = get_game_rule_prompt(agent["name"], board_height, board_width, position) + f"""
        # Task:
        Based on the current state of the game, give a feedback on the first trial's reasoning and action.

        #First trial's reasoning and action:
        You initially chose {refined_action} at first trial by the reason: '{refined_reasoning}'.

        # Output:
        Provide your feedback for the move and the action in the following JSON format:
        ```
        {{
            "feedback": string  // This is the feedback for the initially selected action and reasoning.
        }}
        ```
        """

        parsed_content = get_agent_response(agent, feedback_prompt, system_prompt="You are a rational game analyzer.")
        feedback = parsed_content.get("feedback", "No feedback received.") if parsed_content else "No feedback received."

        refine_prompt = get_game_rule_prompt(agent["name"], board_height, board_width, position) + f"""
        You initially chose {refined_action} at first trial by the reason: '{refined_reasoning}'.
        You received feedback on your action and reasoning: {feedback}

        # Task:
        Based on the current state of the game and the feedback, refine your reasoning and action. And finally, decide the move on this turn.

        # Output:
        Provide your reasoning for the move and the action in the following JSON format:
        ```
        {{
            "reasoning":  string  // Explain why you chose the move.
            "action": [row, col] // The square you will move to [row, col]
        }}
        """

        refined_reasoning, refined_action = get_valid_move_with_retry(
            agent, refine_prompt, "You are a skilled Corner Queen player.", position, board_width, board_height,
            fallback_reason="Fallback: random valid queen move"
        )

        if not refined_action:
            return refined_reasoning, []

    return refined_reasoning, refined_action

def get_move_with_debate_2(agent1, agent2, judge_agent, position, board_width, board_height, debate_rounds=3):
    prompt_initial = get_game_rule_prompt(agent1["name"], board_height, board_width, position) \
            + "\n" + get_standard_decision_instruction() 
    reasoning, action = get_valid_move_with_retry(
        agent1,
        prompt_initial,
        "You are a skilled Corner Queen player and debating the best action.",
        position,
        board_width,
        board_height,
        fallback_reason="Fallback: Randomly selected a valid queen move."
    )

    current_reasoning = reasoning
    current_action = action
    
    debate_history = f"Game Current State: {prompt_initial}\n Initial Answer: {current_action} with reasoning: {current_reasoning}\n"
    
    for round_num in range(1, debate_rounds + 1):
        prompt_negative = f"{prompt_initial}\n"
        prompt_negative += f"""
        You are {agent2['name']}, playing as the negative side. You disagree with the current answer.
        Your goal is to challenge the current answer and provide reasons why it might be incorrect.
        The current answer is {current_action} with reasoning: '{current_reasoning}'.\n
        """
        prompt_negative += get_standard_decision_instruction()
        negative_reasoning, negative_action = get_valid_move_with_retry(
        agent2,
        prompt_negative,
        "You are negative side. Provide your critique and alternative answer as a JSON formatted snippet with keys 'reasoning' and 'action'.",
        position,
        board_width,
        board_height,
        fallback_reason="Fallback: Randomly selected a valid queen move."
        )
        
        prompt_affirmative = f"{prompt_initial}\n"
        prompt_affirmative += f"""
        You are {agent1['name']}, playing as the affirmative side. You originally provided the answer {current_action} with reasoning: '{current_reasoning}'.
        After hearing the negative evaluation which suggested an alternative answer of {negative_action} with reasoning: '{negative_reasoning}', please refine or confirm your answer.\n
        """
        prompt_affirmative += get_standard_decision_instruction()
        refined_reasoning, refined_action = get_valid_move_with_retry(
        agent1,
        prompt_affirmative,
        "You are affirmative side. Refine your answer based on the negative feedback.",
        position,
        board_width,
        board_height,
        fallback_reason="Fallback: Randomly selected a valid queen move."
        )
        
        debate_history += f"Round {round_num}:\n"
        debate_history += f"Negative: action={negative_action}, reasoning='{negative_reasoning}'\n"
        debate_history += f"Affirmative: action={refined_action}, reasoning='{refined_reasoning}'\n"
        
        current_action = refined_action
        current_reasoning = refined_reasoning
        
        if negative_action == current_action:
            return current_reasoning, current_action
    
    judge_prompt = f"""
    You are a moderator. There have been two debaters (affirmative and negative) discussing the best move in a game of Corner Queen. 
    Debate Topic: decide which square to move to [row, col].
    
    Debate History:
    {debate_history}
    
    Based on the debate history, please decide which debater's answer is correct. Provide your final judgment by outputting a JSON formatted snippet with keys "reasoning" (explain your decision) and "action" (the chosen move as an an list [row, col]).
    """
    final_reasoning, final_action = get_valid_move_with_retry(
        judge_agent,
        judge_prompt,
        "You are a moderator. Evaluate both sides' arguments and decide which one is correct.",
        position,
        board_width,
        board_height,
        fallback_reason="Fallback: Randomly selected a valid queen move."
        )

    return final_reasoning, final_action

def get_move_with_debate(agent1, agent2, position, board_width, board_height):
    """
    Debate-style move selection in Corner Queen.
    Returns (reasoning, action).
    """
    initial_moves = {}
    initial_reasonings = {}

    for i, agent in enumerate([agent1, agent2]):
        prompt = get_game_rule_prompt(agent["name"], board_height, board_width, position) \
            + "\n" + get_standard_decision_instruction() 
        reasoning, action = get_valid_move_with_retry(
            agent,
            prompt,
            "You are a skilled Corner Queen player and debating the best action.",
            position,
            board_width,
            board_height,
            fallback_reason="Fallback: Randomly selected a valid queen move."
        )
        initial_moves[f"agent{i+1}"] = action
        initial_reasonings[f"agent{i+1}"] = reasoning

    for _ in range(debate_rounds):
        for i, agent in enumerate([agent1, agent2]):
            other_i = 1 - i
            agent_key = f"agent{i+1}"
            other_key = f"agent{other_i+1}"

            prompt = get_game_rule_prompt(agent["name"], board_height, board_width, position) + "\n" + f"""
            # Task:
            Based on the current state of the game, decide which square to move to [row, col] on this turn.
                 
            You initially chose {initial_moves[agent_key]} items at first trial by the reason: '{initial_reasonings[agent_key]}'.\n
            Other agent argues that you have to choose move as: {initial_moves[other_key]} by the reason: {initial_reasonings[other_key]}.\n
            Considering the other's opinion, refine or confirm your move.\n

            # Output:
            Provide your reasoning for the move and the action in the following JSON format:
            ```    
            {{
                "reasoning":  string  // Explain why you chose the move.
                "action": [row, col] // The square you will move to [row, col]
            }}
            ```
            """

            reasoning, action = get_valid_move_with_retry(
                agent,
                prompt,
                "You are a skilled Corner Queen player and debating the best action.",
                position,
                board_width,
                board_height,
                fallback_reason="Fallback: random valid queen move."
            )
            initial_moves[agent_key] = action
            initial_reasonings[agent_key] = reasoning

        all_actions = [tuple(move) for move in initial_moves.values()]
        if len(set(all_actions)) == 1:
            return initial_reasonings["agent1"], initial_moves["agent1"]

    action_counts = Counter(tuple(move) for move in initial_moves.values())
    most_common_move = list(action_counts.most_common(1)[0][0])
    return initial_reasonings["agent1"], most_common_move

def get_move_dreamad(agent1, agent2, position, board_width, board_height):
    initial_moves = {}
    initial_reasonings = {}

    r, c = position
    game_prompt = get_game_rule_prompt(agent1["name"], board_height, board_width, position) + f"""
    # Task:
    Based on the current state of the game, decide which square to move to [row, col].
    """
    extract_prompt = f"""Below is a game description. Extract key information.

    {game_prompt}

    ### Format Response as:
    ```
    {{
    "game_definition": string // What is the definition of this game?.
    "winning_condition": string // How to win the game.
    "move_constraints": string // What amoves are allowed.
    }}
    ```
    """
    parsed_content = get_agent_response(agent1, extract_prompt, "You are a game theorist and strategist. Output your answer as a valid JSON with keys 'game_definition', 'winning_condition', and 'move_constraints'.", temperature=0.1)
    game_definition = parsed_content.get("game_definition")
    winning_condition = parsed_content.get("winning_condition")
    move_constraints = parsed_content.get("move_constraints")

    strategy_prompt = f"""
    Based on the game information below, derive the **optimal strategy**.

    **Game:** {game_definition}  
    **Winning Condition:** {winning_condition}  
    **Move Constraints:** {move_constraints}

    ### Format Response as:
    ```
    {{
    "state_evaluation": string // How to assess the game state.
    "winning_strategy": sstring  // Winning strategy in this turn to win this game.
    "endgame_tactics": string // Best strategy in a near-win situation.
    }}
    ```
    """
    parsed_content = get_agent_response(agent1, strategy_prompt, "You are a game theorist and strategist. ", temperature=0.1)
    state_eval = parsed_content.get("state_evaluation")
    winning_strat = parsed_content.get("winning_strategy")
    endgame = parsed_content.get("endgame_tactics")

    for idx, agent in enumerate([agent1, agent2]):
        base_prompt = f"""
        Refine the initial game prompt to improve decision-making based on the Game and Strategy.
        ##Initial prompt:
        {game_prompt}

        **Game:** {game_definition}  
        **Strategy:**  
        - State Evaluation: {state_eval}  
        - Winning Strategy: {winning_strat}  
        - Endgame Tactics: {endgame}  

        ### Instructions:
        1. The new prompt must **clearly guide decision-making**.
        2. It should **force the model to prioritize winning moves**.
        3. Language should be **direct, logical, and assertive**.
        4. Do NOT include the answer — only refine the prompt.
        5. Do NOT define the format of the output.

        ### Format Response as:
        ```
        {{
        "optimized_prompt": string // The refined prompt that clearly directs decision-making.
        }}
        ```
        """
        parsed_content = get_agent_response(agent, base_prompt, "You are a good game theorist and strategist. Output your answer as a valid JSON with keys 'optimized_prompt'.", temperature=0.7)
        opt_prompt = parsed_content.get("optimized_prompt")

        full_prompt = opt_prompt + f"""
        **Current State:**  
        - Board size: {board_height} x {board_width}  
        - Queen's position: [row={r}, col={c}]

        ### Instructions:
        1. **If a winning move exists, take it immediately.**  
        2. **Otherwise, follow optimal move principles.**  
        3. Justify your move using the extracted strategy.

        # Output:
        Provide your reasoning for the move and the action in the following JSON format:
        ```
        {{
            "reasoning":  string  // Explain why you chose the move.
            "action": [row, col] // The square you will move to [row, col]
        }}
        ```
        """

        response = get_agent_response(agent, full_prompt, "You are a game theorist and strategist. Output your answer as a valid JSON with keys 'reasoning' and 'action'.")
        action = response.get("action")
        reasoning = response.get("reasoning")

        key = f"agent{idx + 1}"
        initial_moves[key] = action
        initial_reasonings[key] = reasoning
        if idx == 0:
            prompt1 = opt_prompt
        else:
            prompt2 = opt_prompt

    for _ in range(debate_rounds):
        for idx, agent in enumerate([agent1, agent2]):
            other_key = "agent2" if idx == 0 else "agent1"
            self_key = f"agent{idx + 1}"
            used_prompt = prompt1 if idx == 0 else prompt2

            debate_prompt = used_prompt + f"""

            You initially chose {initial_moves[self_key]} by the reason: '{initial_reasonings[self_key]}'.
            Other agent argues: {initial_moves[other_key]} by the reason: '{initial_reasonings[other_key]}'.
            Considering the other's opinion and your strategy, refine or confirm your move.

            ### Instructions:
            1. **If a winning move exists, take it immediately.**
            2. **Otherwise, follow optimal move principles.**
            3. Justify your move using the extracted strategy.

            # Output:
            Provide your reasoning for the move and the action in the following JSON format:
            ```
            {{
                "reasoning":  string  // Explain why you chose the move.
                "action": [row, col] // The square you will move to [row, col]
            }}
            ```"""

            parsed = get_agent_response(agent, debate_prompt, "You are a skilled Corner Queen player and debating the best move. Output your answer as a valid JSON with keys 'action'.")
            reasoning = parsed.get("reasoning")
            action = parsed.get("action")
            initial_reasonings[self_key] = reasoning
            initial_moves[self_key] = action

        all_actions = [tuple(v) for v in initial_moves.values() if isinstance(v, list)]
        if len(set(all_actions)) == 1:
            return initial_reasonings["agent1"], list(all_actions[0])
        
    action_counts = Counter(tuple(action) for action in initial_moves.values() if isinstance(action, list))
    most_common = list(action_counts.most_common(1)[0][0])
    return initial_reasonings["agent1"], most_common

def play_corner_queen_game(board_w, board_h, initial_position, verbose=False, file_handle=None):
    queen_pos = initial_position.copy()
    turn = 0

    while True:
        current_agent = agents[turn % 2]
        other_agent = agents[(turn + 1) % 2]
        pm = current_agent["prompting_method"]
        if pm == "simple":
            reasoning, move = get_move(current_agent, queen_pos, board_w, board_h)
        elif pm == "basic":
            reasoning, move = get_basic_move(current_agent, queen_pos, board_w, board_h)
        elif pm == "self_consistency":
            reasoning, move = get_consistent_move(current_agent, queen_pos, board_w, board_h, num_responses=self_consistency_count)
        elif pm == "self_reflection":
            reasoning, move = get_move_with_reflection(current_agent, queen_pos, board_w, board_h)
        elif pm == "debate":
            reasoning, move = get_move_with_debate(current_agent, current_agent, queen_pos, board_w, board_h)
        elif pm == "debate2":
            reasoning, move = get_move_with_debate_2(current_agent, current_agent, current_agent, queen_pos, board_w, board_h)
        elif pm == "dreamad":
            reasoning, move = get_move_dreamad(current_agent, current_agent, queen_pos, board_w, board_h)
        else:
            reasoning, move = get_basic_move(current_agent, queen_pos, board_w, board_h)

        if len(move) == 2 and is_valid_move(queen_pos, move, board_w, board_h):
            apply_move(queen_pos, move)
        else:
            fallback = get_random_valid_move(queen_pos, board_w, board_h)
            if fallback:
                reasoning = "Fallback random move"
                apply_move(queen_pos, fallback)
            else:
                if verbose:
                    print(f"{current_agent['name']} has no valid moves. Stalemate.")
                if file_handle:
                    print(f"{current_agent['name']} has no valid moves. Stalemate.", file=file_handle)
                return None

        if verbose:
            print(f"{current_agent['name']} ({current_agent['model']}) => reasoning: {reasoning}, action: {move}, pos: {queen_pos}")
        if file_handle:
            print(f"{current_agent['name']} ({current_agent['model']}) => reasoning: {reasoning}, action: {move}, pos: {queen_pos}", file=file_handle)

        if is_in_corner(queen_pos, board_w, board_h):
            if verbose:
                print(f"{current_agent['name']} wins by reaching corner!")
            if file_handle:
                print(f"{current_agent['name']} wins by reaching corner!", file=file_handle)
            return current_agent["name"]

        turn += 1

def simulate_corner_queen_games(num_games, board_width, board_height, initial_position, verbose=False):

    win_counts = {agent["name"]: 0 for agent in agents}
    with open(f'/YourPath/{args.agent1_model}_{args.agent1_prompt}_{args.agent2_model}_{args.agent2_prompt}.txt', 'a') as f:
        for g in range(num_games):
            if verbose:
                print(f"\n--- Starting Corner Queen Game {g+1} ---")
            print(f"\n--- Starting Corner Queen Game {g+1} ---", file=f)

            winner = play_corner_queen_game(board_width, board_height, initial_position, verbose=verbose, file_handle=f)
            if winner:
                win_counts[winner] += 1

        if verbose:
            print("\nGame Results:")
        print("\nGame Results:", file=f)
        for ag in agents:
            wcount = win_counts[ag["name"]]
            rate = 100.0 * wcount / num_games
            if verbose:
                print(f"{ag['name']} Win Rate: {rate:.2f}%  ({wcount}/{num_games})")
            print(f"{ag['name']} Win Rate: {rate:.2f}%  ({wcount}/{num_games})", file=f)

if __name__ == "__main__":
    simulate_corner_queen_games(
        num_games=num_games,
        board_width=board_width,
        board_height=board_height,
        initial_position=initial_queen_position,
        verbose=True
    )
