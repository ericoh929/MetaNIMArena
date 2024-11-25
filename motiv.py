import openai
import os
import random
from collections import Counter
import argparse


parser = argparse.ArgumentParser(description='arguments for training')

parser.add_argument('--agent1_model',     type=str,   default=None, help='model')
parser.add_argument('--agent2_model',     type=str,   default=None, help='model')
parser.add_argument('--agent1_prompt',     type=str,   default='basic', help='prompt_method')
parser.add_argument('--agent2_prompt',     type=str,   default='basic', help='prompt_method')

args = parser.parse_args()

# Set up your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize the game parameters
total_items = 21  # Total items in the pile (e.g., 21)
max_take = 3  # Maximum items that can be taken per turn
num_games = 50  # Number of games to play
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

    Based on the current state of the game, decide how many items you will take. Provide only the integer number of items, between 1 and {max_take}, with no extra text or symbols. For example,\n
    2
    """
    response = openai.ChatCompletion.create(
        model=agent["model"],
        messages=[
            {"role": "system", "content": "You are a skilled Nim player."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=5,
        temperature=0.7
    )
    try:
        move = int(response.choices[0].message['content'].strip())
        if 1 <= move <= max_take:
            return move
    except ValueError:
        pass
    return random.randint(1, max_take)

def random_move():
    
    return random.randint(1, max_take)

# Function for self-consistency: generate multiple responses and choose the most common move
def get_consistent_move(agent, remaining_items, num_responses):
    prompt = f"""
    You are {agent['name']} in a game of Nim. There are {remaining_items} items remaining in the pile.
    You can take between 1 and {max_take} items on your turn. The goal is to win by taking all remaining items on your turn, leaving no items for your opponent.

    Based on the current state of the game, decide how many items you will take. Provide only the integer number of items, between 1 and {max_take}, with no extra text or symbols. For example,\n
    2
    """
    moves = []
    for _ in range(num_responses):
        response = openai.ChatCompletion.create(
            model=agent["model"],
            messages=[
                {"role": "system", "content": "You are a skilled Nim player."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=5,
            temperature=0.7
        )
        try:
            move = int(response.choices[0].message['content'].strip())
            if 1 <= move <= max_take:
                moves.append(move)
        except ValueError:
            pass
    if not moves:
        print(f"Warning: No valid responses from {agent['name']} (self-consistency). Falling back to random move.")
        return random.randint(1, max_take)
    most_common_move = Counter(moves).most_common(1)[0][0]
    return most_common_move

# Function for self-reflection prompting
def get_move_with_reflection(agent, remaining_items):
    prompt_initial = f"""
    You are {agent['name']} in a game of Nim. There are {remaining_items} items remaining in the pile.
    You can take between 1 and {max_take} items on your turn. The goal is to win by taking all remaining items on your turn, leaving no items for your opponent.

    Based on the current state of the game, decide how many items you will take. Provide only the integer number of items, between 1 and {max_take}, with no extra text or symbols. For example,\n
    2
    """
    response_initial = openai.ChatCompletion.create(
        model=agent["model"],
        messages=[{"role": "system", "content": "You are a skilled Nim player."}, {"role": "user", "content": prompt_initial}],
        max_tokens=5,
        temperature=0.7
    )
    try:
        initial_move = int(response_initial.choices[0].message['content'].strip())
    except ValueError:
        initial_move = random.randint(1, max_take)

    reflection_prompt = f"""
    You are {agent['name']} in a game of Nim. There are {remaining_items} items remaining in the pile.
    You can take between 1 and {max_take} items on your turn. The goal is to win by taking all remaining items on your turn, leaving no items for your opponent.
    You initially chose {initial_move} items at first trial.

    Reflect: is this optimal?
    If yes, confirm; if not, suggest a better move.

    Provide only the integer number of items, between 1 and {max_take}, with no extra text or symbols. For example,\n
    2
    """
    reflection_response = openai.ChatCompletion.create(
        model=agent["model"],
        messages=[{"role": "system", "content": "You are a skilled Nim player. Reflect on your move."}, {"role": "user", "content": reflection_prompt}],
        max_tokens=10,
        temperature=0.7
    )
    reflection_text = reflection_response.choices[0].message['content'].strip()
    if "confirm" in reflection_text.lower():
        return initial_move
    else:
        # Improved: Ensure reflection response suggests valid moves
        suggested_move = [int(s) for s in reflection_text.split() if s.isdigit() and 1 <= int(s) <= max_take]
        if not suggested_move:
            print(f"Warning: Invalid reflection response from {agent['name']}. Keeping initial move.")
            return initial_move
        return suggested_move[0]


def self_play_debate(agent1, agent2, remaining_items, n_step_lookahead):
    moves = []  # Track each agent's moves for each lookahead step
    
    for step in range(1, n_step_lookahead + 1):
        # Agent 1's move
        prompt_agent1 = f"""
        You are {agent1['name']} in a game of Nim. There are {remaining_items} items remaining in the pile.
        You can take between 1 and {max_take} items on your turn. The goal is to win by taking all remaining items on your turn, leaving no items for your opponent.

        Based on the current state of the game, decide how many items you will take for your first move. Provide only the number of items as an integer.
        """

        response_agent1 = openai.ChatCompletion.create(
            model=agent1["model"],
            messages=[
                {"role": "system", "content": "You are a skilled Nim player. Provide only the number of items to take."},
                {"role": "user", "content": prompt_agent1}
            ],
            max_tokens=5,
            temperature=0.7
        )

        try:
            agent1_move = int(response_agent1.choices[0].message['content'].strip())
            if not (1 <= agent1_move <= max_take):
                raise ValueError
        except ValueError:
            agent1_move = random.randint(1, max_take)

        moves.append((agent1["name"], agent1_move))
        remaining_items -= agent1_move

        # Check if game ends with Agent 1's move
        if remaining_items <= 0:
            return agent1_move  # Agent 1 wins if no items remain

        # Agent 2's simulated response
        prompt_agent2 = f"""
        You are {agent2['name']} in a game of Nim. There are {remaining_items} items remaining after {agent1['name']} took {agent1_move} items.
        You can take between 1 and {max_take} items on your turn. The goal is to win by taking all remaining items on your turn, leaving no items for your opponent.

        Based on the current state of the game, decide how many items you will take for your first move. Provide only the number of items as an integer.
        """

        response_agent2 = openai.ChatCompletion.create(
            model=agent1["model"],
            messages=[
                {"role": "system", "content": "You are a skilled Nim player. Provide only the number of items to take."},
                {"role": "user", "content": prompt_agent2}
            ],
            max_tokens=5,
            temperature=0.7
        )

        try:
            agent2_move = int(response_agent2.choices[0].message['content'].strip())
            if not (1 <= agent2_move <= max_take):
                raise ValueError
        except ValueError:
            agent2_move = random.randint(1, max_take)

        moves.append((agent2["name"], agent2_move))
        remaining_items -= agent2_move

        # Check if game ends with Agent 2's move
        if remaining_items <= 0:
            return agent1_move  # Agent 1's initial move if Agent 2 would win

    # Final decision for Agent 1 based on the full n-step lookahead sequence
    move_sequence_str = "; ".join([f"{name} took {move} items" for name, move in moves])
    final_prompt_agent1 = f"""
    You are {agent1['name']} in a game of Nim. Initially, you considered taking {agent1_move} items with {remaining_items + agent2_move} items remaining.
    After {n_step_lookahead} steps, the predicted move sequence was: {move_sequence_str}.
    Based on this simulated sequence, decide how many items you will take for the best outcome. Provide only the integer number of items, between 1 and {max_take}.
    """

    final_response_agent1 = openai.ChatCompletion.create(
        model=agent1["model"],
        messages=[
            {"role": "system", "content": "You are a skilled Nim player. Reflect on the sequence of moves and refine your action."},
            {"role": "user", "content": final_prompt_agent1}
        ],
        max_tokens=5,
        temperature=0.7
    )

    try:
        agent1_final_move = int(final_response_agent1.choices[0].message['content'].strip())
        if not (1 <= agent1_final_move <= max_take):
            raise ValueError
    except ValueError:
        agent1_final_move = random.randint(1, max_take)
    
    return agent1_final_move

# def self_play_debate(agent1, agent2, remaining_items):
#     prompt_agent1 = f"""
#     You are {agent1['name']} in a game of Nim. There are {remaining_items} items remaining in the pile.
#     You can take between 1 and {max_take} items on your turn. The goal is to win by taking all remaining items on your turn, leaving no items for your opponent.

#     Based on the current state of the game, decide how many items you will take. Provide only the integer number of items, between 1 and {max_take}, with no extra text or symbols. For example,\n
#     2
#     """
#     response_agent1 = openai.ChatCompletion.create(
#         model=agent1["model"],
#         messages=[{"role": "system", "content": "You are a skilled Nim player."},
#                   {"role": "user", "content": prompt_agent1}],
#         max_tokens=5,
#         temperature=0
#     )
#     try:
#         agent1_initial_move = int(response_agent1.choices[0].message['content'].strip())
#         if not (1 <= agent1_initial_move <= max_take):
#             raise ValueError
#     except ValueError:
#         agent1_initial_move = random.randint(1, max_take)

#     simulated_remaining_items = remaining_items - agent1_initial_move
#     prompt_agent2 = f"""
#     You are {agent1['name']} in a game of Nim. There are {simulated_remaining_items} items remaining after {agent1['name']} took {agent1_initial_move} items.
#     You can take between 1 and {max_take} items on your turn. The goal is to win by taking all remaining items on your turn, leaving no items for your opponent.

#     Based on the current state of the game, decide how many items you will take. Provide only the integer number of items, between 1 and {max_take}, with no extra text or symbols. For example,\n
#     2
#     """
#     response_agent2 = openai.ChatCompletion.create(
#         model=agent1["model"],
#         messages=[{"role": "system", "content": "You are a skilled Nim player."},
#                   {"role": "user", "content": prompt_agent2}],
#         max_tokens=5,
#         temperature=0
#     )
#     try:
#         agent2_move = int(response_agent2.choices[0].message['content'].strip())
#         if not (1 <= agent2_move <= max_take):
#             raise ValueError
#     except ValueError:
#         agent2_move = random.randint(1, max_take)

#     final_prompt_agent1 = f"""
#     You are {agent1['name']} in a game of Nim. There are {remaining_items} items remaining in the pile.
#     You can take between 1 and {max_take} items on your turn. The goal is to win by taking all remaining items on your turn, leaving no items for your opponent.

#     Initially, you considered taking {agent1_initial_move} items with {remaining_items} items remaining.
#     You expect {agent2['name']} to respond by taking {agent2_move} items.
    
#     Based on the current state of the game and {agent2['name']}'s policy, decide again how many items you will take. Provide only the integer number of items, between 1 and {max_take}, with no extra text or symbols. For example,\n
#     2
#     """
#     final_response_agent1 = openai.ChatCompletion.create(
#         model=agent1["model"],
#         messages=[{"role": "system", "content": "You are a skilled Nim player. Reflect on your move considering opponent move."},
#                   {"role": "user", "content": final_prompt_agent1}],
#         max_tokens=5,
#         temperature=0
#     )
#     try:
#         agent1_final_move = int(final_response_agent1.choices[0].message['content'].strip())
#         if not (1 <= agent1_final_move <= max_take):
#             raise ValueError
#     except ValueError:
#         agent1_final_move = random.randint(1, max_take)
    
#     return agent1_final_move


# Debate function for two agents
def get_move_with_debate(agent1, agent2, remaining_items):
    initial_moves = {}
    for agent in [agent1, agent2]:
        prompt = f"""
        You are {agent['name']} in a game of Nim. There are {remaining_items} items remaining in the pile.
        You can take between 1 and {max_take} items on your turn. The goal is to win by taking all remaining items on your turn, leaving no items for your opponent.
        
        Based on the current state of the game, decide how many items you will take. Provide only the integer number of items, between 1 and {max_take}, with no extra text or symbols. For example,\n
        2
        """
        response = openai.ChatCompletion.create(
            model=agent1["model"],
            messages=[{"role": "system", "content": "You are a skilled Nim player and debating the best move."}, {"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0.7
        )
        try:
            move = int(response.choices[0].message['content'].strip().split()[0])
            initial_moves[agent["name"]] = move
        except (ValueError, IndexError):
            initial_moves[agent["name"]] = random.randint(1, max_take)

    # If agents agree, return move
    if len(set(initial_moves.values())) == 1:
        return list(initial_moves.values())[0]

    # Otherwise, conduct debate rounds to reach consensus
    for _ in range(debate_rounds):
        agent_moves_str = "\n".join(f"{name}: {move}" for name, move in initial_moves.items())
        refined_moves = {}
        for agent in [agent1, agent2]:
            prompt = f"""
            You are {agent['name']} in a game of Nim in a debate with {remaining_items} items left.
            You can take between 1 and {max_take} items on your turn. The goal is to win by taking all remaining items on your turn, leaving no items for your opponent.

            Other moves: {agent_moves_str}. Refine or confirm your move.

            Based on the current state of the game, and other moves, refine or confirm your move. Provide only the integer number of items, between 1 and {max_take}, with no extra text or symbols. For example,\n
            2
            """
            response = openai.ChatCompletion.create(
                model=agent1["model"],
                messages=[{"role": "system", "content": "You are a skilled Nim player. Refine your move in debate."}, {"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=0.7
            )
            try:
                move = int(response.choices[0].message['content'].strip().split()[0])
                refined_moves[agent["name"]] = move
            except (ValueError, IndexError):
                refined_moves[agent["name"]] = initial_moves[agent["name"]]

        initial_moves = refined_moves
        if len(set(initial_moves.values())) == 1:
            return list(initial_moves.values())[0]

    return Counter(initial_moves.values()).most_common(1)[0][0]  # Use most common if no consensus

# Game function with all methods integrated
def play_nim_game(total_items, max_take, verbose=False):
    current_items = total_items
    turn = 0
    while current_items > 0:
        current_agent = agents[turn % 2]
        other_agent = agents[(turn + 1) % 2]

        if current_agent["prompting_method"] == "self_consistency":
            move = get_consistent_move(current_agent, current_items, self_consistency_count)
        # elif current_agent["prompting_method"] == "n_step_lookahead":
        #     move = get_move_with_n_step_lookahead(current_agent, other_agent, current_items)
        elif current_agent["prompting_method"] == "self_reflection":
            move = get_move_with_reflection(current_agent, current_items)
        elif current_agent["prompting_method"] == "debate":
            move = get_move_with_debate(current_agent, other_agent, current_items)
        elif current_agent["prompting_method"] == "self_play_debate":
            move = self_play_debate(current_agent, other_agent, current_items, n_step_lookahead)
        elif current_agent["prompting_method"] == "random":
            move = random_move()
        else:
            # print(f"Error: Unknown prompting method '{current_agent['prompting_method']}' for {current_agent['name']}. Using basic move.")
            move = get_basic_move(current_agent, current_items)

        if verbose:
            print(f"{current_agent['name']} ({current_agent['model']}) takes {move} items. Items remaining: {current_items - move}")
        current_items -= move

        if current_items <= 0:
            if verbose:
                print(f"{current_agent['name']} ({current_agent['model']}) wins!")
            return current_agent["name"]

        turn += 1

# Run the simulation
def simulate_games(num_games, total_items, max_take):
    win_counts = {agent["name"]: 0 for agent in agents}
    for game_num in range(num_games):
        print(f"\nStarting Game {game_num + 1}")
        winner = play_nim_game(total_items, max_take, verbose=True)
        win_counts[winner] += 1
    print("\nGame Results:")
    for agent in agents:
        win_rate = (win_counts[agent["name"]] / num_games) * 100
        print(f"{agent['name']} Win Rate: {win_rate:.2f}% ({win_counts[agent['name']]} wins out of {num_games})")

simulate_games(num_games, total_items, max_take)