import os
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

parser.add_argument('--agent1_model',     type=str,   default=None, help='model')
parser.add_argument('--agent2_model',     type=str,   default=None, help='model')
parser.add_argument('--agent1_prompt',     type=str,   default='basic', help='prompt_method')
parser.add_argument('--agent2_prompt',     type=str,   default='basic', help='prompt_method')
parser.add_argument('--num_games',     type=int,   default='50', help='number of games')
parser.add_argument('--temperature',     type=float,   default='0.7', help='temperature')

args = parser.parse_args()

total_items = 20  # Total items in the pile (e.g., 21)
num_games = args.num_games  # Number of games to play
num_refine = 3
self_consistency_count = 10  # Number of responses to use for self-consistency
n_step_lookahead = args.look_ahead  # Number of lookahead steps for n-step opponent modeling
debate_rounds = 3  # Maximum number of debate rounds
spc_temperature = args.temperature

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

def get_move(agent, remaining_items, max_take, last_taken):
    if last_taken is None:
        prompt = f"""
#Game Role:\n You are {agent['name']}, a participant in a simple Fibonacci game.\n\n
#Objective:\n Your goal is to win the game by taking all remaining stones on your turn, leaving no items for your opponent. The person who takes the last stones wins.\n\n
#Game Rule:\n 1. There is a single pile of stones.\n
2. Players take turns to take stones.\n
3. The first player can take any number of stones, but not all the stones in the first move.\n
4. On subsequent turns, the number of stones a player can take must be at least 1 and at most twice the number of stones the previous player took.\n
5. The player who takes the last stone wins the game.\n\n
#Current State:\n There are {remaining_items} stones remaining in the pile.\n
You can take between 1 and {max_take-1} stones on your turn, where {max_take-1} = min(2 × {last_taken}, {remaining_items-1}).\n\n

#Task:\nYou are the first player. Based on the current state of the game, decide how many items you will take (between 1 and {remaining_items-1}) on this turn.\n\n

The output should be a markdown code snippet formatted in the following schema, including the leading and trailing \\`\\`\\`json" and "\\`\\`\\`":\n\n```\n{{\n\t"action": integer  // This is an action you take based on the reasoning. Only provide integer between 1 and {max_take-1}.\n}}
    """
    else:
        prompt = f"""
#Game Role:\n You are {agent['name']}, a participant in a simple Fibonacci game.\n\n
#Objective:\n Your goal is to win the game by taking all remaining stones on your turn, leaving no items for your opponent. The person who takes the last stones wins.\n\n
#Game Rule:\n 1. There is a single pile of stones.\n
2. Players take turns to take stones.\n
3. The first player can take any number of stones, but not all the stones in the first move.\n
4. On subsequent turns, the number of stones a player can take must be at least 1 and at most twice the number of stones the previous player took.\n
5. The player who takes the last stone wins the game.\n\n
#Current State:\n There are {remaining_items} stones remaining in the pile.\n
The last player took {last_taken} stones.\n
You can take between 1 and {max_take} stones on your turn, where {max_take} = min(2 × {last_taken}, {remaining_items}).\n\n

#Task:\nBased on the current state of the game, decide how many items you will take (between 1 and {max_take}) on this turn.\n\n

The output should be a markdown code snippet formatted in the following schema, including the leading and trailing \\`\\`\\`json" and "\\`\\`\\`":\n\n```\n{{\n\t"action": integer  // This is an action you take based on the reasoning. Only provide integer between 1 and {max_take}.\n}}
"""
        
    parsed_content = get_agent_response(agent, prompt, system_prompt="You are a skilled Fibonacci player.")
    action = parsed_content.get("action")

    return "None", action


def get_basic_move(agent, remaining_items, max_take, last_taken):
    if last_taken is None:
        prompt = f"""
#Game Role:\n You are {agent['name']}, a participant in a simple Fibonacci game.\n\n
#Objective:\n Your goal is to win the game by taking all remaining stones on your turn, leaving no items for your opponent. The person who takes the last stones wins.\n\n
#Game Rule:\n 1. There is a single pile of stones.\n
2. Players take turns to take stones.\n
3. The first player can take any number of stones, but not all the stones in the first move.\n
4. On subsequent turns, the number of stones a player can take must be at least 1 and at most twice the number of stones the previous player took.\n
5. The player who takes the last stone wins the game.\n\n
#Current State:\n There are {remaining_items} stones remaining in the pile.\n
You can take between 1 and {max_take-1} stones on your turn, where {max_take-1} = min(2 × {last_taken}, {remaining_items-1}).\n\n

#Task:\nYou are the first player. Based on the current state of the game, decide how many items you will take (between 1 and {remaining_items-1}) on this turn.\n\n

The output should be a markdown code snippet formatted in the following schema, including the leading and trailing \\`\\`\\`json" and "\\`\\`\\`":\n\n```\n{{\n\t"reasoning": string  // This is the reasons for the action\n\t"action": integer  // This is an action you take based on the reasoning. Only provide integer between 1 and {max_take-1}.\n}}
    """
    else:
        prompt = f"""
#Game Role:\n You are {agent['name']}, a participant in a simple Fibonacci game.\n\n
#Objective:\n Your goal is to win the game by taking all remaining stones on your turn, leaving no items for your opponent. The person who takes the last stones wins.\n\n
#Game Rule:\n 1. There is a single pile of stones.\n
2. Players take turns to take stones.\n
3. The first player can take any number of stones, but not all the stones in the first move.\n
4. On subsequent turns, the number of stones a player can take must be at least 1 and at most twice the number of stones the previous player took.\n
5. The player who takes the last stone wins the game.\n\n
#Current State:\n There are {remaining_items} stones remaining in the pile.\n
The last player took {last_taken} stones.\n
You can take between 1 and {max_take} stones on your turn, where {max_take} = min(2 × {last_taken}, {remaining_items}).\n\n

#Task:\nBased on the current state of the game, decide how many items you will take (between 1 and {max_take}) on this turn.\n\n

The output should be a markdown code snippet formatted in the following schema, including the leading and trailing \\`\\`\\`json" and "\\`\\`\\`":\n\n```\n{{\n\t"reasoning": string  // This is the reasons for the action\n\t"action": integer  // This is an action you take based on the reasoning. Only provide integer between 1 and {max_take}.\n}}
"""
        
    parsed_content = get_agent_response(agent, prompt, system_prompt="You are a skilled Fibonacci player.")
    reasoning = parsed_content.get("reasoning")
    action = parsed_content.get("action")

    return reasoning, action

def get_consistent_move(agent, remaining_items, num_responses, max_take, last_taken):
    if last_taken is None:
        prompt = f"""
        #Game Role:\n You are {agent['name']}, a participant in a simple Fibonacci game.\n\n
        #Objective:\n Your goal is to win the game by taking all remaining stones on your turn, leaving no items for your opponent. The person who takes the last stones wins.\n\n
        #Game Rule:\n 1. There is a single pile of stones.\n
        2. Players take turns to take stones.\n
        3. The first player can take any number of stones, but not all the stones in the first move.\n
        4. On subsequent turns, the number of stones a player can take must be at least 1 and at most twice the number of stones the previous player took.\n
        5. The player who takes the last stone wins the game.\n\n
        #Current State:\n There are {remaining_items} stones remaining in the pile.\n
        You can take between 1 and {max_take-1} stones on your turn, where {max_take-1} = min(2 × {last_taken}, {remaining_items-1}).\n\n

        #Task:\nYou are the first player. Based on the current state of the game, decide how many items you will take (between 1 and {remaining_items-1}) on this turn.\n\n

        The output should be a markdown code snippet formatted in the following schema, including the leading and trailing \\`\\`\\`json" and "\\`\\`\\`":\n\n```\n{{\n\t"reasoning": string  // This is the reasons for the action\n\t"action": integer  // This is an action you take based on the reasoning. Only provide integer between 1 and {max_take-1}.\n}}
            """
    else:
        prompt = f"""
        #Game Role:\n You are {agent['name']}, a participant in a simple Fibonacci game.\n\n
        #Objective:\n Your goal is to win the game by taking all remaining stones on your turn, leaving no items for your opponent. The person who takes the last stones wins.\n\n
        #Game Rule:\n 1. There is a single pile of stones.\n
        2. Players take turns to take stones.\n
        3. The first player can take any number of stones, but not all the stones in the first move.\n
        4. On subsequent turns, the number of stones a player can take must be at least 1 and at most twice the number of stones the previous player took.\n
        5. The player who takes the last stone wins the game.\n\n
        #Current State:\n There are {remaining_items} stones remaining in the pile.\n
        The last player took {last_taken} stones.\n
        You can take between 1 and {max_take} stones on your turn, where {max_take} = min(2 × {last_taken}, {remaining_items}).\n\n

        #Task:\nBased on the current state of the game, decide how many items you will take (between 1 and {max_take}) on this turn.\n\n

        The output should be a markdown code snippet formatted in the following schema, including the leading and trailing \\`\\`\\`json" and "\\`\\`\\`":\n\n```\n{{\n\t"reasoning": string  // This is the reasons for the action\n\t"action": integer  // This is an action you take based on the reasoning. Only provide integer between 1 and {max_take}.\n}}
        """
    moves = []

    for _ in range(num_responses):
        parsed_content = get_agent_response(agent, prompt, system_prompt="You are a skilled Fibonacci player.")
        reasoning = parsed_content.get("reasoning")
        action = parsed_content.get("action")

        move = int(action)
        moves.append(move)

    most_common_move = Counter(moves).most_common(1)[0][0]

    return reasoning, most_common_move

def get_consistent_diverse_move(agent, remaining_items, num_responses, max_take, last_taken):
    moves = []

    prompt = f"""
        #Game Role:\n You are {agent['name']}, a participant in a simple Fibonacci game.\n\n
        #Objective:\n Your goal is to win the game by taking all remaining stones on your turn, leaving no items for your opponent. The person who takes the last stones wins.\n\n
        #Game Rule:\n 1. There is a single pile of stones.\n
        2. Players take turns to take stones.\n
        3. On subsequent turns, the number of stones a player can take must be at least 1 and at most twice the number of stones the previous player took.\n
        4. The player who takes the last stone wins the game.\n\n
        #Current State:\n There are {remaining_items} stones remaining in the pile.\n
        You can take between 1 and {max_take} stones on your turn, where {max_take} = min(2 × {last_taken}, {remaining_items}).\n\n
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

    for _ in range(num_responses):
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

    {{
    "optimized_prompt": "string", // The refined prompt that clearly directs decision-making. }}
    """
        parsed_content = get_agent_response(agent, final_prompt, system_prompt="You are a game theorist and strategist.", temperature=spc_temperature)

        optimized_prompt = parsed_content.get("optimized_prompt")

        one_new_prompt = f"""{optimized_prompt}\n

        **Current State:**  
        - There are {remaining_items} items left.  

        ### Instructions:
        1. **If a winning move exists, take it immediately.**  
        2. **Otherwise, follow optimal move principles.**  
        3. Justify your move using the extracted strategy.

        ### Format Response as:
        {{
        "reasoning": "string", // Explanation of the move based on the strategy.
        "action": integer // This is an action you take based on the reasoning. Only provide integer between 1 and {max_take}.}}
        """

        parsed_content = get_agent_response(agent, one_new_prompt, system_prompt="You are a skilled Fibonacci player.", temperature=1.0)
        reasoning = parsed_content.get("reasoning")
        action = parsed_content.get("action")

        move = int(action)
        moves.append(move)

    most_common_move = Counter(moves).most_common(1)[0][0]

    return reasoning, most_common_move

def get_move_with_reflection(agent, remaining_items, max_take, last_taken):
    if last_taken is None:
        prompt_initial = f"""
        #Game Role:\n You are {agent['name']}, a participant in a simple Fibonacci game.\n\n
        #Objective:\n Your goal is to win the game by taking all remaining stones on your turn, leaving no items for your opponent. The person who takes the last stones wins.\n\n
        #Game Rule:\n 1. There is a single pile of stones.\n
        2. Players take turns to take stones.\n
        3. The first player can take any number of stones, but not all the stones in the first move.\n
        4. On subsequent turns, the number of stones a player can take must be at least 1 and at most twice the number of stones the previous player took.\n
        5. The player who takes the last stone wins the game.\n\n
        #Current State:\n There are {remaining_items} stones remaining in the pile.\n
        You can take between 1 and {max_take-1} stones on your turn, where {max_take-1} = min(2 × {last_taken}, {remaining_items-1}).\n\n

        #Task:\nYou are the first player. Based on the current state of the game, decide how many items you will take (between 1 and {remaining_items-1}) on this turn.\n\n

        The output should be a markdown code snippet formatted in the following schema, including the leading and trailing \\`\\`\\`json" and "\\`\\`\\`":\n\n```\n{{\n\t"reasoning": string  // This is the reasons for the action\n\t"action": integer  // This is an action you take based on the reasoning. Only provide integer between 1 and {max_take-1}.\n}}
            """
    else:
        prompt_initial = f"""
        #Game Role:\n You are {agent['name']}, a participant in a simple Fibonacci game.\n\n
        #Objective:\n Your goal is to win the game by taking all remaining stones on your turn, leaving no items for your opponent. The person who takes the last stones wins.\n\n
        #Game Rule:\n 1. There is a single pile of stones.\n
        2. Players take turns to take stones.\n
        3. The first player can take any number of stones, but not all the stones in the first move.\n
        4. On subsequent turns, the number of stones a player can take must be at least 1 and at most twice the number of stones the previous player took.\n
        5. The player who takes the last stone wins the game.\n\n
        #Current State:\n There are {remaining_items} stones remaining in the pile.\n
        The last player took {last_taken} stones.\n
        You can take between 1 and {max_take} stones on your turn, where {max_take} = min(2 × {last_taken}, {remaining_items}).\n\n

        #Task:\nBased on the current state of the game, decide how many items you will take (between 1 and {max_take}) on this turn.\n\n

        The output should be a markdown code snippet formatted in the following schema, including the leading and trailing \\`\\`\\`json" and "\\`\\`\\`":\n\n```\n{{\n\t"reasoning": string  // This is the reasons for the action\n\t"action": integer  // This is an action you take based on the reasoning. Only provide integer between 1 and {max_take}.\n}}
        """

    parsed_content = get_agent_response(agent, prompt_initial, system_prompt="You are a skilled Fibonacci player.")

    initial_reasoning = parsed_content.get("reasoning")
    action = parsed_content.get("action")

    initial_move = int(action)

    for k in range(num_refine):
        if last_taken is None:
            feedback_prompt = f"""
            #Game Role:\n You are {agent['name']}, a participant in a simple Fibonacci game.\n\n
            #Objective:\n Your goal is to win the game by taking all remaining stones on your turn, leaving no items for your opponent. The person who takes the last stones wins.\n\n
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
            #Game Role:\n You are {agent['name']}, a participant in a simple Fibonacci game.\n\n
            #Objective:\n Your goal is to win the game by taking all remaining stones on your turn, leaving no items for your opponent. The person who takes the last stones wins.\n\n
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
        
        parsed_content = get_agent_response(agent, feedback_prompt, system_prompt="You are a skilled Fibonacci player.")
        feedback = parsed_content.get("feedback")

        if last_taken is None:
            refine_prompt = f"""
            #Game Role:\n You are {agent['name']}, a participant in a simple Fibonacci game.\n\n
            #Objective:\n Your goal is to win the game by taking all remaining stones on your turn, leaving no items for your opponent. The person who takes the last stones wins.\n\n
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

            The output should be a markdown code snippet formatted in the following schema, including the leading and trailing \\`\\`\\`json" and "\\`\\`\\`":\n\n```\n{{\n\t"reasoning": string  // This is the reasons for the action\n\t"action": integer  // This is an action you take based on the reasoning. Only provide integer between 1 and {max_take-1}.\n}}
                """
        else:
            refine_prompt = f"""
            #Game Role:\n You are {agent['name']}, a participant in a simple Fibonacci game.\n\n
            #Objective:\n Your goal is to win the game by taking all remaining stones on your turn, leaving no items for your opponent. The person who takes the last stones wins.\n\n
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

            The output should be a markdown code snippet formatted in the following schema, including the leading and trailing \\`\\`\\`json" and "\\`\\`\\`":\n\n```\n{{\n\t"reasoning": string  // This is the reasons for the action\n\t"action": integer  // This is an action you take based on the reasoning. Only provide integer between 1 and {max_take}.\n}}
            """

        parsed_content = get_agent_response(agent, refine_prompt, system_prompt="You are a skilled Fibonacci player.")
        refined_reasoning = parsed_content.get("reasoning")
        refined_action = parsed_content.get("action")

        if initial_move == int(refined_action):
            return refined_reasoning, refined_action
        else:
            initial_move = refined_action
            initial_reasoning = refined_reasoning

    return refined_reasoning, refined_action

def get_move_with_diverse_reflection(agent, remaining_items, max_take, last_taken):
    prompt = f"""
        #Game Role:\n You are {agent['name']}, a participant in a simple Fibonacci game.\n\n
        #Objective:\n Your goal is to win the game by taking all remaining stones on your turn, leaving no items for your opponent. The person who takes the last stones wins.\n\n
        #Game Rule:\n 1. There is a single pile of stones.\n
        2. Players take turns to take stones.\n
        3. On subsequent turns, the number of stones a player can take must be at least 1 and at most twice the number of stones the previous player took.\n
        4. The player who takes the last stone wins the game.\n\n
        #Current State:\n There are {remaining_items} stones remaining in the pile.\n
        You can take between 1 and {max_take} stones on your turn, where {max_take} = min(2 × {last_taken}, {remaining_items}).\n\n
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

    {{
    "optimized_prompt": "string", // The refined prompt that clearly directs decision-making. }}
    """
    parsed_content = get_agent_response(agent, final_prompt, system_prompt="You are a game theorist and strategist.", temperature=spc_temperature)

    optimized_prompt = parsed_content.get("optimized_prompt")

    one_new_prompt = f"""{optimized_prompt}\n

    **Current State:**  
    - There are {remaining_items} items left.  

    ### Instructions:
    1. **If a winning move exists, take it immediately.**  
    2. **Otherwise, follow optimal move principles.**  
    3. Justify your move using the extracted strategy.

    ### Format Response as:
    {{
    "reasoning": "string", // Explanation of the move based on the strategy.
    "action": integer // This is an action you take based on the reasoning. Only provide integer between 1 and {max_take}.}}
    """

    parsed_content = get_agent_response(agent, one_new_prompt, system_prompt="You are a skilled Fibonacci player.")

    initial_reasoning = parsed_content.get("reasoning")
    action = parsed_content.get("action")

    initial_move = int(action)

    for k in range(num_refine):
        if last_taken is None:
            feedback_prompt = f"""
            #Game Role:\n You are {agent['name']}, a participant in a simple Fibonacci game.\n\n
            #Objective:\n Your goal is to win the game by taking all remaining stones on your turn, leaving no items for your opponent. The person who takes the last stones wins.\n\n
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
            #Game Role:\n You are {agent['name']}, a participant in a simple Fibonacci game.\n\n
            #Objective:\n Your goal is to win the game by taking all remaining stones on your turn, leaving no items for your opponent. The person who takes the last stones wins.\n\n
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
        
        parsed_content = get_agent_response(agent, feedback_prompt, system_prompt="You are a skilled Fibonacci player.")
        feedback = parsed_content.get("feedback")

        if last_taken is None:
            refine_prompt = f"""
            #Game Role:\n You are {agent['name']}, a participant in a simple Fibonacci game.\n\n
            #Objective:\n Your goal is to win the game by taking all remaining stones on your turn, leaving no items for your opponent. The person who takes the last stones wins.\n\n
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

            The output should be a markdown code snippet formatted in the following schema, including the leading and trailing \\`\\`\\`json" and "\\`\\`\\`":\n\n```\n{{\n\t"reasoning": string  // This is the reasons for the action\n\t"action": integer  // This is an action you take based on the reasoning. Only provide integer between 1 and {max_take-1}.\n}}
                """
        else:
            refine_prompt = f"""
            #Game Role:\n You are {agent['name']}, a participant in a simple Fibonacci game.\n\n
            #Objective:\n Your goal is to win the game by taking all remaining stones on your turn, leaving no items for your opponent. The person who takes the last stones wins.\n\n
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

            The output should be a markdown code snippet formatted in the following schema, including the leading and trailing \\`\\`\\`json" and "\\`\\`\\`":\n\n```\n{{\n\t"reasoning": string  // This is the reasons for the action\n\t"action": integer  // This is an action you take based on the reasoning. Only provide integer between 1 and {max_take}.\n}}
            """

        parsed_content = get_agent_response(agent, refine_prompt, system_prompt="You are a skilled Fibonacci player.")
        refined_reasoning = parsed_content.get("reasoning")
        refined_action = parsed_content.get("action")

        if initial_move == int(refined_action):
            return refined_reasoning, refined_action
        else:
            initial_move = refined_action
            initial_reasoning = refined_reasoning

    return refined_reasoning, refined_action


def get_move_with_debate(agent1, agent2, remaining_items, max_take, last_taken):
    initial_moves = {}
    initial_reasonings = {}
    i = 0
    for agent in [agent1, agent2]:
        if last_taken is None:
            prompt = f"""
#Game Role:\n You are {agent['name']}, a participant in a simple Fibonacci game.\n\n
#Objective:\n Your goal is to win the game by taking all remaining stones on your turn, leaving no items for your opponent. The person who takes the last stones wins.\n\n
#Game Rule:\n 1. There is a single pile of stones.\n
2. Players take turns to take stones.\n
3. The first player can take any number of stones, but not all the stones in the first move.\n
4. On subsequent turns, the number of stones a player can take must be at least 1 and at most twice the number of stones the previous player took.\n
5. The player who takes the last stone wins the game.\n\n
#Current State:\n There are {remaining_items} stones remaining in the pile.\n
You can take between 1 and {max_take-1} stones on your turn, where {max_take-1} = min(2 × {last_taken}, {remaining_items-1}).\n\n

#Task:\nYou are the first player. Based on the current state of the game, decide how many items you will take (between 1 and {remaining_items-1}) on this turn.\n\n

The output should be a markdown code snippet formatted in the following schema, including the leading and trailing \\`\\`\\`json" and "\\`\\`\\`":\n\n```\n{{\n\t"reasoning": string  // This is the reasons for the action\n\t"action": integer  // This is an action you take based on the reasoning. Only provide integer between 1 and {max_take-1}.\n}}
    """
        else:
            prompt = f"""
#Game Role:\n You are {agent['name']}, a participant in a simple Fibonacci game.\n\n
#Objective:\n Your goal is to win the game by taking all remaining stones on your turn, leaving no items for your opponent. The person who takes the last stones wins.\n\n
#Game Rule:\n 1. There is a single pile of stones.\n
2. Players take turns to take stones.\n
3. The first player can take any number of stones, but not all the stones in the first move.\n
4. On subsequent turns, the number of stones a player can take must be at least 1 and at most twice the number of stones the previous player took.\n
5. The player who takes the last stone wins the game.\n\n
#Current State:\n There are {remaining_items} stones remaining in the pile.\n
The last player took {last_taken} stones.\n
You can take between 1 and {max_take} stones on your turn, where {max_take} = min(2 × {last_taken}, {remaining_items}).\n\n

#Task:\nBased on the current state of the game, decide how many items you will take (between 1 and {max_take}) on this turn.\n\n

The output should be a markdown code snippet formatted in the following schema, including the leading and trailing \\`\\`\\`json" and "\\`\\`\\`":\n\n```\n{{\n\t"reasoning": string  // This is the reasons for the action\n\t"action": integer  // This is an action you take based on the reasoning. Only provide integer between 1 and {max_take}.\n}}
"""
        
        parsed_content = get_agent_response(agent, prompt, system_prompt="You are a skilled Fibonacci player and debating the best move.")

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
                if last_taken is None:
                    prompt = f"""
#Game Role:\n You are {agent['name']}, a participant in a simple Fibonacci game.\n\n
#Objective:\n Your goal is to win the game by taking all remaining stones on your turn, leaving no items for your opponent. The person who takes the last stones wins.\n\n
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

The output should be a markdown code snippet formatted in the following schema, including the leading and trailing \\`\\`\\`json" and "\\`\\`\\`":\n\n```\n{{\n\t"reasoning": string  // This is the reasons for the action\n\t"action": integer  // This is an action you take based on the reasoning. Only provide integer between 1 and {max_take-1}.\n}}
"""
                else:
                    prompt = f"""
#Game Role:\n You are {agent['name']}, a participant in a simple Fibonacci game.\n\n
#Objective:\n Your goal is to win the game by taking all remaining stones on your turn, leaving no items for your opponent. The person who takes the last stones wins.\n\n
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

The output should be a markdown code snippet formatted in the following schema, including the leading and trailing \\`\\`\\`json" and "\\`\\`\\`":\n\n```\n{{\n\t"reasoning": string  // This is the reasons for the action\n\t"action": integer  // This is an action you take based on the reasoning. Only provide integer between 1 and {max_take}.\n}}
"""
            if i == 1:
                if last_taken is None:
                    prompt = f"""
#Game Role:\n You are {agent['name']}, a participant in a simple Fibonacci game.\n\n
#Objective:\n Your goal is to win the game by taking all remaining stones on your turn, leaving no items for your opponent. The person who takes the last stones wins.\n\n
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

The output should be a markdown code snippet formatted in the following schema, including the leading and trailing \\`\\`\\`json" and "\\`\\`\\`":\n\n```\n{{\n\t"reasoning": string  // This is the reasons for the action\n\t"action": integer  // This is an action you take based on the reasoning. Only provide integer between 1 and {max_take-1}.\n}}
"""
                else:
                    prompt = f"""
#Game Role:\n You are {agent['name']}, a participant in a simple Fibonacci game.\n\n
#Objective:\n Your goal is to win the game by taking all remaining stones on your turn, leaving no items for your opponent. The person who takes the last stones wins.\n\n
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

The output should be a markdown code snippet formatted in the following schema, including the leading and trailing \\`\\`\\`json" and "\\`\\`\\`":\n\n```\n{{\n\t"reasoning": string  // This is the reasons for the action\n\t"action": integer  // This is an action you take based on the reasoning. Only provide integer between 1 and {max_take}.\n}}
"""
            parsed_content = get_agent_response(agent, prompt, system_prompt="You are a skilled Fibonacci player and debating the best move.")

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

    return initial_reasoning, Counter(initial_moves.values()).most_common(1)[0][0]  


def get_move_with_debate_2(agent1, agent2, judge_agent, remaining_items, max_take, last_taken, debate_rounds=3):
    prompt_initial = f"""
    #Game Role:\n You are a participant in a simple Fibonacci game.\n\n
    #Objective:\n Your goal is to win the game by taking all remaining stones on your turn, leaving no items for your opponent. The person who takes the last stones wins.\n\n
    #Game Rule:\n 1. There is a single pile of stones.\n
    2. Players take turns to take stones.\n
    3. On subsequent turns, the number of stones a player can take must be at least 1 and at most twice the number of stones the previous player took.\n
    4. The player who takes the last stone wins the game.\n\n
    #Current State:\n There are {remaining_items} stones remaining in the pile.\n
    You can take between 1 and {max_take} stones on your turn, where {max_take} = min(2 × {last_taken}, {remaining_items}).\n\n
    #Task:\nBased on the current state of the game, decide how many items you will take (between 1 and {max_take}) on this turn.\n\n
    
    Output your answer as a JSON formatted snippet with keys "reasoning" (string) and "action" (integer).
    """
    initial_result = get_agent_response(agent1, prompt_initial, system_prompt="You are a skilled Fibonacci player.", temperature=0.7)
    initial_reasoning = initial_result.get("reasoning")
    initial_action = initial_result.get("action")
    current_reasoning = initial_result.get("reasoning")
    current_action = int(initial_result.get("action"))
    debate_history = f"Game Current State: {prompt_initial}\n Initial Answer: {initial_action} with reasoning: {initial_reasoning}\n"
    
    for round_num in range(1, debate_rounds + 1):
        prompt_negative = f"{prompt_initial}\n"
        prompt_negative += f"""
        You are {agent2['name']}, playing as the negative side. You disagree with the current answer.
        Your goal is to challenge the current answer and provide reasons why it might be incorrect.
        There are {remaining_items} items remaining in the pile, and the current answer is {current_action} with reasoning: '{current_reasoning}'.
        
        #Task:
        Provide your updated reasoning and an alternative answer (an integer between 1 and {max_take}).
        
        Output your answer as a JSON formatted snippet with keys "reasoning" and "action".
        """
        result_neg = get_agent_response(agent2, prompt_negative, 
                                        system_prompt="You are negative side. Provide your critique and alternative answer as a JSON formatted snippet with keys 'reasoning' and 'action'.", 
                                        temperature=0.7)
        negative_action = result_neg.get("action")
        negative_reasoning = result_neg.get("reasoning")
        prompt_affirmative = f"{prompt_initial}\n"
        prompt_affirmative += f"""
        You are {agent1['name']}, playing as the affirmative side. You originally provided the answer {current_action} with reasoning: '{current_reasoning}'.
        After hearing the negative evaluation which suggested an alternative answer of {negative_action} with reasoning: '{negative_reasoning}', please refine or confirm your answer.
        Provide your updated reasoning and answer (an integer between 1 and {max_take}).
        
        Output your answer as a JSON formatted snippet with keys "reasoning" and "action".
        """
        result_affirm = get_agent_response(agent1, prompt_affirmative, 
                                           system_prompt="You are affirmative side. Refine your answer based on the negative feedback.", 
                                           temperature=0.7)
        refined_action = result_affirm.get("action")
        refined_reasoning = result_affirm.get("reasoning")
        
        debate_history += f"Round {round_num}:\n"
        debate_history += f"Negative: action={negative_action}, reasoning='{negative_reasoning}'\n"
        debate_history += f"Affirmative: action={refined_action}, reasoning='{refined_reasoning}'\n"
        
        current_action = int(refined_action)
        current_reasoning = refined_reasoning
        
        if int(negative_action) == current_action:
            return current_reasoning, current_action
    
    judge_prompt = f"""
    You are a moderator. There have been two debaters (affirmative and negative) discussing the best move in a game of Nim variants. 
    Debate Topic: Taking items from a single pile with {remaining_items} items remaining, where you can take between 1 and {max_take} items.
    
    Debate History:
    {debate_history}
    
    Based on the debate history, please decide which debater's answer is correct. Provide your final judgment by outputting a JSON formatted snippet with keys "reasoning" (explain your decision) and "action" (the chosen move as an integer).
    """
    judge_result = get_agent_response(judge_agent, judge_prompt, 
                                      system_prompt="You are a moderator. Evaluate both sides' arguments and decide which one is correct.", 
                                      temperature=0.7)
    final_reasoning = judge_result.get("reasoning")
    final_action = int(judge_result.get("action"))
    
    return final_reasoning, int(final_action)

def get_move_dreamad(agent1, agent2, remaining_items, max_take, last_taken):
    initial_moves = {}
    initial_reasonings = {}
    i = 0
    for agent in [agent1]:
        prompt = f"""
        #Game Role:\n You are {agent['name']}, a participant in a simple Fibonacci game.\n\n
        #Objective:\n Your goal is to win the game by taking all remaining stones on your turn, leaving no items for your opponent. The person who takes the last stones wins.\n\n
        #Game Rule:\n 1. There is a single pile of stones.\n
        2. Players take turns to take stones.\n
        3. On subsequent turns, the number of stones a player can take must be at least 1 and at most twice the number of stones the previous player took.\n
        4. The player who takes the last stone wins the game.\n\n
        #Current State:\n There are {remaining_items} stones remaining in the pile.\n
        You can take between 1 and {max_take} stones on your turn, where {max_take} = min(2 × {last_taken}, {remaining_items}).\n\n
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
    
        ### Instructions:
        1. **If a winning move exists, take it immediately.**  
        2. **Otherwise, follow optimal move principles.**  
        3. Justify your move using the extracted strategy.

        ### Format Response as:
        {{
        "reasoning": "string", // Explanation of the move based on the strategy.
        "action": integer // This is an action you take based on the reasoning. Only provide integer between 1 and {max_take}.}}
        """
        one_parsed_content = get_agent_response(agent, one_new_prompt, system_prompt="You are a game theorist and strategist.")
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


    for k in range(debate_rounds):
        i = 0
        for agent in [agent1, agent2]:
            
            if i == 0:
                prompt = f"""
                {one_prompt}\n

                You initially chose {initial_moves['agent1']} items at first trial by the reason: '{initial_reasonings['agent1']}'.\n
                Other agent argues that you have to choose move as: {initial_moves['agent2']} by the reason: {initial_reasonings['agent2']}.\n
                Considering the other's opinion and your strategy, refine or confirm your move.\n

                ### Format Response as:
                {{
                "reasoning": "string", // Explanation of the move based on the strategy.
                "action": integer // This is an action you take based on the reasoning. Only provide integer between 1 and {max_take}.
                }}
                """
            if i == 1:
                prompt = f"""
                {two_prompt}\n

                You initially chose {initial_moves['agent2']} items at first trial by the reason: '{initial_reasonings['agent2']}'.\n
                Other agent argues that you have to choose move as: {initial_moves['agent1']} by the reason: {initial_reasonings['agent1']}.\n
                Considering the other's opinion and your strategy, refine or confirm your move.\n

                ### Format Response as:
                {{
                "reasoning": "string", // Explanation of the move based on the strategy.
                "action": integer // This is an action you take based on the reasoning. Only provide integer between 1 and {max_take}.
                }}
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
                
            i += 1
        initial_moves['agent1'] = a0_action
        initial_reasonings['agent1'] = a0_reasoning
        initial_moves['agent2'] = a1_action
        initial_reasonings['agent2'] = a1_reasoning

        if len(set(initial_moves.values())) == 1:
            return initial_reasoning, initial_action

    return initial_reasoning, Counter(initial_moves.values()).most_common(1)[0][0]  


def play_fibonacci_nim_game(total_items, verbose=False):
    with open(f'/YourPath/{args.agent1_model}_{args.agent1_prompt}_{args.agent2_model}_{args.agent2_prompt}.txt', 'a') as f:
        current_items = total_items
        turn = 0
        last_taken = None 

        while current_items > 0:
            current_agent = agents[turn % 2]
            other_agent = agents[(turn + 1) % 2]
            max_take = current_items if last_taken is None else min(2 * last_taken, current_items)

            if current_agent["prompting_method"] == "self_consistency":
                reasoning, move = get_consistent_move(current_agent, current_items, self_consistency_count, max_take, last_taken)
            elif current_agent["prompting_method"] == "diverse_consistency":
                reasoning, move = get_consistent_diverse_move(current_agent, current_items, self_consistency_count, max_take, last_taken)
            elif current_agent["prompting_method"] == "self_reflection":
                reasoning, move = get_move_with_reflection(current_agent, current_items, max_take, last_taken)
            elif current_agent["prompting_method"] == "diverse_reflection":
                reasoning, move = get_move_with_diverse_reflection(current_agent, current_items, max_take, last_taken)
            elif current_agent["prompting_method"] == "debate":
                reasoning, move = get_move_with_debate(current_agent, current_agent, current_items, max_take, last_taken)
            elif current_agent["prompting_method"] == "debate2":
                reasoning, move = get_move_with_debate_2(current_agent, current_agent, current_agent, current_items, max_take, last_taken)
            elif current_agent["prompting_method"] == "dreamad":
                reasoning, move = get_move_dreamad(current_agent, current_agent, current_items, max_take, last_taken)
            elif current_agent["prompting_method"] == "simple":
                reasoning, move = get_move(current_agent, current_items, max_take, last_taken)
            elif current_agent["prompting_method"] == "basic":
                reasoning, move = get_basic_move(current_agent, current_items, max_take, last_taken)
            else:
                print("Error: set the prompting methods")
                return None

            if verbose:
                print(f"Reasoning: {reasoning}\nAction: {move}", file=f)
                print(f"{current_agent['name']} ({current_agent['model']}) takes {move} items. Items remaining: {current_items - move}", file=f)

            current_items -= move
            last_taken = move  
            if current_items <= 0:
                if verbose:
                    print(f"{current_agent['name']} ({current_agent['model']}) wins!", file=f)
                return current_agent["name"]

            turn += 1

def simulate_fibonacci_nim_games(num_games, total_items):
    win_counts = {agent["name"]: 0 for agent in agents}
    with open(f'/YourPath/{args.agent1_model}_{args.agent1_prompt}_{args.agent2_model}_{args.agent2_prompt}.txt', 'a') as f:
        for game_num in range(num_games):
            print(f"\nStarting Game {game_num + 1}", file=f)
            print(f"\nStarting Game {game_num + 1}")
            winner = play_fibonacci_nim_game(total_items, verbose=True)
            win_counts[winner] += 1

        print("\nGame Results:", file=f)
        for agent in agents:
            win_rate = (win_counts[agent["name"]] / num_games) * 100
            print(f"{agent['name']} Win Rate: {win_rate:.2f}% ({win_counts[agent['name']]} wins out of {num_games})", file=f)

simulate_fibonacci_nim_games(num_games, total_items)