import openai
import os
import random
from collections import Counter
import argparse
import json
import re
from openai import OpenAI

import os
import google.generativeai as genai
genai.configure(api_key=os.environ['GEMINI_API_KEY'])

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"),)



parser = argparse.ArgumentParser(description='arguments for training')

parser.add_argument('--agent1_model',     type=str,   default=None, help='model')
parser.add_argument('--agent2_model',     type=str,   default=None, help='model')
parser.add_argument('--agent1_prompt',     type=str,   default='basic', help='prompt_method')
parser.add_argument('--agent2_prompt',     type=str,   default='basic', help='prompt_method')

args = parser.parse_args()

# Set up your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize the game parameters
total_items = 31  # Total items in the pile (e.g., 21)
max_take = 3  # Maximum items that can be taken per turn
num_games = 50  # Number of games to play
num_refine = 3
self_consistency_count = 10  # Number of responses to use for self-consistency
n_step_lookahead = 3  # Number of lookahead steps for n-step opponent modeling
debate_rounds = 3  # Maximum number of debate rounds

# Define agents with their respective models and prompting methods
agents = [
    {"name": "Agent 1", "model": args.agent1_model, "prompting_method": args.agent1_prompt},
    {"name": "Agent 2", "model": args.agent2_model, "prompting_method": args.agent2_prompt}
]



# Function for basic move (single-response without consistency or modeling)
def get_basic_move(agent, remaining_items):
    prompt = f"""
    You are {agent['name']} in a game of Nim. There are {remaining_items} items remaining in the pile.
    You can take between 1 and {max_take} items on your turn. The goal is to win by taking all remaining items on your turn, leaving no items for your opponent.

    Based on the current state of the game, decide how many items you will take between 1 and {max_take}.
    The output should be a markdown code snippet formatted in the following schema, including the leading and trailing \\`\\`\\`json" and "\\`\\`\\`":\n\n```\n{{\n\t"reasoning": string  // This is the reasons for the action\n\t"action": integer  // This is an action you take. Only provide integer.\n}}
    """

    if agent["model"] == 'gemini-1.5-flash':

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
    else:
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

    parsed_content = json.loads(parsed_content_with_braces)

    reasoning = parsed_content.get("reasoning")
    action = parsed_content.get("action")

    return reasoning, action

# Function for self-consistency: generate multiple responses and choose the most common move
def get_consistent_move(agent, remaining_items, num_responses):
    prompt = f"""
    You are {agent['name']} in a game of Nim. There are {remaining_items} items remaining in the pile.
    You can take between 1 and {max_take} items on your turn. The goal is to win by taking all remaining items on your turn, leaving no items for your opponent.

    Based on the current state of the game, decide how many items you will take between 1 and {max_take}.
    The output should be a markdown code snippet formatted in the following schema, including the leading and trailing \\`\\`\\`json" and "\\`\\`\\`":\n\n```\n{{\n\t"reasoning": string  // This is the reasons for the action\n\t"action": integer  // This is an action you take. Only provide integer.\n}}
    """
    moves = []

    if agent["model"] == 'gemini-1.5-flash':
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
    else:
        for _ in range(num_responses):
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

# Extracted content including { }
    parsed_content_with_braces = matches_with_braces.group(0) if matches_with_braces else None

    # print(parsed_content_with_braces)

    # content_trimmed = content[3:-3].strip()
    # print('content trimmed::', content_trimmed)
    # parsed_content = json.loads(content_trimmed)
    parsed_content = json.loads(parsed_content_with_braces)

    # Extract reasoning and action
    reasoning = parsed_content.get("reasoning")
    action = parsed_content.get("action")
    move = int(action)
    moves.append(move)
    most_common_move = Counter(moves).most_common(1)[0][0]

    return reasoning, most_common_move

# Function for self-reflection prompting
def get_move_with_reflection(agent, remaining_items):
    prompt_initial = f"""
    You are {agent['name']} in a game of Nim. There are {remaining_items} items remaining in the pile.
    You can take between 1 and {max_take} items on your turn. The goal is to win by taking all remaining items on your turn, leaving no items for your opponent.

    Based on the current state of the game, decide how many items you will take between 1 and {max_take}.
    The output should be a markdown code snippet formatted in the following schema, including the leading and trailing \\`\\`\\`json" and "\\`\\`\\`":\n\n```\n{{\n\t"reasoning": string  // This is the reasons for the action\n\t"action": integer  // This is an action you take. Only provide integer.\n}}
    """

    if agent["model"] == 'gemini-1.5-flash':

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

    else:
        response_initial = client.chat.completions.create(
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
        
        content = response_initial.choices[0].message.content

    matches_with_braces = re.search(r'\{.*?\}', content, re.DOTALL)

    parsed_content_with_braces = matches_with_braces.group(0) if matches_with_braces else None
    parsed_content = json.loads(parsed_content_with_braces)

    initial_reasoning = parsed_content.get("reasoning")
    action = parsed_content.get("action")

    initial_move = int(action)

    for k in range(num_refine):

        feedback_prompt = f"""
        You are {agent['name']} in a game of Nim. There are {remaining_items} items remaining in the pile.
        You can take between 1 and {max_take} items on your turn. The goal is to win by taking all remaining items on your turn, leaving no items for your opponent.
        You initially chose {initial_move} items at first trial by the reason: '{initial_reasoning}'.

        Based on the current state of the game, give a feedback on the first trial's reasoning and action.

        The output should be a markdown code snippet formatted in the following schema, including the leading and trailing \\`\\`\\`json" and "\\`\\`\\`":\n\n```\n{{\n\t"feedback": string  // This is the feedback for the selected action and reasoning\n}}
        """

        if agent["model"] == 'gemini-1.5-flash':

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

        else:
            feedback_response = client.chat.completions.create(
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
            
            content = feedback_response.choices[0].message.content

        matches_with_braces = re.search(r'\{.*?\}', content, re.DOTALL)

        parsed_content_with_braces = matches_with_braces.group(0) if matches_with_braces else None
        parsed_content = json.loads(parsed_content_with_braces)

        feedback = parsed_content.get("feedback")

        refine_prompt = f"""
        You are {agent['name']} in a game of Nim. There are {remaining_items} items remaining in the pile.
        You can take between 1 and {max_take} items on your turn. The goal is to win by taking all remaining items on your turn, leaving no items for your opponent.
        You initially chose {initial_move} items at first trial by the reason: '{initial_reasoning}'.\n\n
        You recieved feedback on your action and reasoning: {feedback}

        Based on the current state of the game and the feedback, refine your reasoning and action. 
        Finally, Decide how many items you will take between 1 and {max_take}.
        The output should be a markdown code snippet formatted in the following schema, including the leading and trailing \\`\\`\\`json" and "\\`\\`\\`":\n\n```\n{{\n\t"reasoning": string  // This is the reasons for the action\n\t"action": integer  // This is an action you take. Only provide integer.\n}}
        """

        if agent["model"] == 'gemini-1.5-flash':

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

        else:
            response_refined = client.chat.completions.create(
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
            
            content = response_refined.choices[0].message.content

        matches_with_braces = re.search(r'\{.*?\}', content, re.DOTALL)

        parsed_content_with_braces = matches_with_braces.group(0) if matches_with_braces else None
        parsed_content = json.loads(parsed_content_with_braces)

        refined_reasoning = parsed_content.get("reasoning")
        refined_action = parsed_content.get("action")

        if initial_move == int(refined_action):
            return refined_reasoning, refined_action
        else:
            initial_move = refined_action
            initial_reasoning = refined_reasoning

    return refined_reasoning, refined_action


def self_play_debate(agent1, agent2, remaining_items, n_step_lookahead):
    initial_remaining_items = remaining_items
    moves = []  # Track each agent's moves for each lookahead step
    planning = ''
    for step in range(1, n_step_lookahead + 1):
        # Agent 1's move
        state = f"""You are {agent1['name']} in a game of Nim. There are {remaining_items} items remaining in the pile."""
        prompt_agent1 = f"""
        {state}
        You can take between 1 and {max_take} items on your turn. The goal is to win by taking all remaining items on your turn, leaving no items for your opponent.

        Based on the current state of the game, decide how many items you will take between 1 and {max_take}.
        The output should be a markdown code snippet formatted in the following schema, including the leading and trailing \\`\\`\\`json" and "\\`\\`\\`":\n\n```\n{{\n\t"reasoning": string  // This is the reasons for the action\n\t"action": integer  // This is an action you take. Only provide integer.\n}}
        """

        if agent1["model"] == 'gemini-1.5-flash':

        # Create the model
            generation_config = {
            "temperature": 0.7,
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

        else:

            response_agent1 = client.chat.completions.create(
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
                temperature=0.7
                )
        
            content = response_agent1.choices[0].message.content

        matches_with_braces = re.search(r'\{.*?\}', content, re.DOTALL)

        parsed_content_with_braces = matches_with_braces.group(0) if matches_with_braces else None
        parsed_content = json.loads(parsed_content_with_braces)

        agent1_reasoning = parsed_content.get("reasoning")
        agent1_action = parsed_content.get("action")

        agent1_move = int(agent1_action)

        moves.append((agent1["name"], agent1_move))
        planning += f'State: {state}\n'
        planning += f'My reasoning: {agent1_reasoning}\n'
        planning += f'My action: {agent1_action}\n\n'
        
        remaining_items -= agent1_move

        # Check if game ends with Agent 1's move
        if remaining_items <= 0:
            return agent1_reasoning, agent1_move  # Agent 1 wins if no items remain

        # Agent 2's simulated response
        state = f"""You are {agent2['name']} in a game of Nim. There are {remaining_items} items remaining in the pile."""
        prompt_agent2 = f"""
        {state}
        You can take between 1 and {max_take} items on your turn. The goal is to win by taking all remaining items on your turn, leaving no items for your opponent.

        Based on the current state of the game, decide how many items you will take between 1 and {max_take}.
        The output should be a markdown code snippet formatted in the following schema, including the leading and trailing \\`\\`\\`json" and "\\`\\`\\`":\n\n```\n{{\n\t"reasoning": string  // This is the reasons for the action\n\t"action": integer  // This is an action you take. Only provide integer.\n}}
        """

        if agent1["model"] == 'gemini-1.5-flash':

            # Create the model
            generation_config = {
            "temperature": 0.7,
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

        else:
            response_agent2 = client.chat.completions.create(
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
                temperature=0.7
                )
        
            content = response_agent2.choices[0].message.content

        matches_with_braces = re.search(r'\{.*?\}', content, re.DOTALL)

        parsed_content_with_braces = matches_with_braces.group(0) if matches_with_braces else None
        parsed_content = json.loads(parsed_content_with_braces)

        agent2_reasoning = parsed_content.get("reasoning")
        agent2_action = parsed_content.get("action")

        agent2_move = int(agent2_action)

        planning += f'State: {state}\n'
        planning += f'Opponent reasoning: {agent2_reasoning}\n'
        planning += f'Opponent action: {agent2_action}\n\n'

        moves.append((agent2["name"], agent2_move))
        remaining_items -= agent2_move
        

        # Check if game ends with Agent 2's move
        # if remaining_items <= 0:
        #     return agent1_move  # Agent 1's initial move if Agent 2 would win

    # Final decision for Agent 1 based on the full n-step lookahead sequence
    move_sequence_str = "; ".join([f"{name} took {move} items" for name, move in moves])
    final_prompt_agent1 = f"""
    You are {agent1['name']} in a game of Nim. There are {initial_remaining_items} items remaining in the pile.
    You can take between 1 and {max_take} items on your turn. The goal is to win by taking all remaining items on your turn, leaving no items for your opponent.

    You simulated several steps ahead by modeling the opponent and the planning result is as below:\n
    Planning: \n{planning} \nAfter {n_step_lookahead} steps, the predicted move sequence was: {move_sequence_str}.

    Based on the current state of the game, opponent's strategy, and simulated planning history, refine your action and decide again how many items you will take between 1 and {max_take} at the first time step when there are  {initial_remaining_items} items remaining in the pile.
    
    The output should be a markdown code snippet formatted in the following schema, including the leading and trailing \\`\\`\\`json" and "\\`\\`\\`":\n\n```\n{{\n\t"reasoning": string  // This is the reasons for the action\n\t"action": integer  // This is an action you take. Only provide integer.\n}}
    """

    if agent1["model"] == 'gemini-1.5-flash':

        # Create the model
        generation_config = {
        "temperature": 0.7,
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
    
    else:
        final_response_agent1 = client.chat.completions.create(
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
                temperature=0.7
                )
        
        content = final_response_agent1.choices[0].message.content

    matches_with_braces = re.search(r'\{.*?\}', content, re.DOTALL)

    parsed_content_with_braces = matches_with_braces.group(0) if matches_with_braces else None
    parsed_content = json.loads(parsed_content_with_braces)

    agent1_reasoning = parsed_content.get("reasoning")
    agent1_action = parsed_content.get("action")

    agent1_final_move = int(agent1_action)
    
    return agent1_reasoning, agent1_final_move


def get_move_with_debate(agent1, agent2, remaining_items):
    initial_moves = {}
    initial_reasonings = {}
    for agent in [agent1, agent2]:
        prompt = f"""
        You are {agent['name']} in a game of Nim. There are {remaining_items} items remaining in the pile.
        You can take between 1 and {max_take} items on your turn. The goal is to win by taking all remaining items on your turn, leaving no items for your opponent.

        Based on the current state of the game, decide how many items you will take between 1 and {max_take}.
        The output should be a markdown code snippet formatted in the following schema, including the leading and trailing \\`\\`\\`json" and "\\`\\`\\`":\n\n```\n{{\n\t"reasoning": string  // This is the reasons for the action\n\t"action": integer  // This is an action you take. Only provide integer.\n}}
        """

        if agent["model"] == 'gemini-1.5-flash':

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

        else:
            response_initial = client.chat.completions.create(
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
        
            content = response_initial.choices[0].message.content

        matches_with_braces = re.search(r'\{.*?\}', content, re.DOTALL)

        parsed_content_with_braces = matches_with_braces.group(0) if matches_with_braces else None
        parsed_content = json.loads(parsed_content_with_braces)

        initial_reasoning = parsed_content.get("reasoning")
        initial_action = parsed_content.get("action")
        initial_moves[agent["name"]] = initial_action
        initial_reasonings[agent["name"]] = initial_reasoning

        
    # If agents agree, return move
    if len(set(initial_moves.values())) == 1:
        return initial_reasoning, list(initial_moves.values())[0]

    # Otherwise, conduct debate rounds to reach consensus
    for _ in range(debate_rounds):
        # agent_moves_str = "\n".join(f"{name}: {move}" for name, move in initial_moves.items())
        # refined_moves = {}
        for agent in [agent1, agent2]:
            others = [a for a in [agent1, agent2] if a != agent]
            other = others[0]
            prompt = f"""
            You are {agent['name']} in a game of Nim. There are {remaining_items} items remaining in the pile.
            You can take between 1 and {max_take} items on your turn. The goal is to win by taking all remaining items on your turn, leaving no items for your opponent.
            You initially chose {initial_moves[agent["name"]]} items at first trial by the reason: '{initial_reasonings[agent["name"]]}'.

            Other agent argues that you have to choose move as: {initial_moves[other["name"]]} by the reason: {initial_reasonings[other["name"]]}. 
            Considering the other's opinion, refine or confirm your move. Decide how many items you will take between 1 and {max_take}.
            The output should be a markdown code snippet formatted in the following schema, including the leading and trailing \\`\\`\\`json" and "\\`\\`\\`":\n\n```\n{{\n\t"reasoning": string  // This is the reasons for the action\n\t"action": integer  // This is an action you take. Only provide integer.\n}}
            """

            if agent["model"] == 'gemini-1.5-flash':

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
            
            else:
                response_initial = client.chat.completions.create(
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
        
                content = response_initial.choices[0].message.content

            matches_with_braces = re.search(r'\{.*?\}', content, re.DOTALL)

            parsed_content_with_braces = matches_with_braces.group(0) if matches_with_braces else None
            parsed_content = json.loads(parsed_content_with_braces)

            initial_reasoning = parsed_content.get("reasoning")
            initial_action = parsed_content.get("action")
            initial_moves[agent["name"]] = initial_action
            initial_reasonings[agent["name"]] = initial_reasoning

        # initial_moves = refined_moves
        if len(set(initial_moves.values())) == 1:
            return initial_reasoning, list(initial_moves.values())[0]

    return initial_reasoning, Counter(initial_moves.values()).most_common(1)[0][0]  # Use most common if no consensus

# Game function with all methods integrated
def play_nim_game(total_items, max_take, verbose=False):
    with open(f'/home/jihwan/NashIP/result/BR31/{args.agent1_model}_{args.agent1_prompt}_{args.agent2_model}_{args.agent2_prompt}.txt', 'a') as f:
        current_items = total_items
        turn = 0
        while current_items > 0:
            current_agent = agents[turn % 2]
            other_agent = agents[(turn + 1) % 2]

            if current_agent["prompting_method"] == "self_consistency":
                reasoning, move = get_consistent_move(current_agent, current_items, self_consistency_count)
            # elif current_agent["prompting_method"] == "n_step_lookahead":
            #     move = get_move_with_n_step_lookahead(current_agent, other_agent, current_items)
            elif current_agent["prompting_method"] == "self_reflection":
                reasoning, move = get_move_with_reflection(current_agent, current_items)
            elif current_agent["prompting_method"] == "debate":
                reasoning, move = get_move_with_debate(current_agent, other_agent, current_items)
            elif current_agent["prompting_method"] == "self_play_debate":
                reasoning, move = self_play_debate(current_agent, other_agent, current_items, n_step_lookahead)
            elif current_agent["prompting_method"] == "basic":
                reasoning, move = get_basic_move(current_agent, current_items)
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
    for game_num in range(num_games):
        print(f"\nStarting Game {game_num + 1}")
        winner = play_nim_game(total_items, max_take, verbose=True)
        win_counts[winner] += 1
    with open(f'/home/jihwan/NashIP/result/BR31/{args.agent1_model}_{args.agent1_prompt}_{args.agent2_model}_{args.agent2_prompt}.txt', 'a') as f:
        print("\nGame Results:", file = f)
        for agent in agents:
            win_rate = (win_counts[agent["name"]] / num_games) * 100
            print(f"{agent['name']} Win Rate: {win_rate:.2f}% ({win_counts[agent['name']]} wins out of {num_games})", file = f)

simulate_games(num_games, total_items, max_take)