import os
import openai
from openai import OpenAI

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"),)
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

import json

from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser


# parser.add_argument('--agent1_model',     type=str,   default=None, help='model')
# parser.add_argument('--agent2_model',     type=str,   default=None, help='model')
# parser.add_argument('--agent1_prompt',     type=str,   default='basic', help='prompt_method')
# parser.add_argument('--agent2_prompt',     type=str,   default='basic', help='prompt_method')

# args = parser.parse_args()

# Set up your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize the game parameters
total_items = 21  # Total items in the pile (e.g., 21)
max_take = 3  # Maximum items that can be taken per turn
num_games = 50  # Number of games to play
self_consistency_count = 5  # Number of responses to use for self-consistency
n_step_lookahead = 3  # Number of lookahead steps for n-step opponent modeling
debate_rounds = 3  # Maximum number of debate rounds

# Define agents with their respective models and prompting methods
# agents = [
#     {"name": "Agent 1", "model": args.agent1_model, "prompting_method": args.agent1_prompt},
#     {"name": "Agent 2", "model": args.agent2_model, "prompting_method": args.agent2_prompt}
# ]


question = """What action would you take if a man chase you now? The output should be a markdown code snippet formatted in the following schema, including the leading and trailing \\`\\`\\`json" and "\\`\\`\\`":\n\n```json\n{{\n\t"reasoning": string  // This is the reasons for the action\n\t"action": string  // This is an action you take\n"""

response = client.chat.completions.create(
    messages=[
        {
            "role": "system",
            "content": "You are a smart man",
        },
        {
            "role": "user",
            "content": f"{question}",
        }
    ],
    model="gpt-3.5-turbo",
    temperature=0.7
)
content = response.choices[0].message.content
print(content[:3])
print('json file', content)

content_trimmed = content[3:-3].strip()
# content_trimmed = content.strip()


print('json file', content_trimmed)

parsed_content = json.loads(content_trimmed)

# Extract reasoning and action
reasoning = parsed_content.get("reasoning")
action = parsed_content.get("action")

print('Reasoning:', reasoning, '\nAction:', action)


# response = openai.ChatCompletion.create(
#         model='gpt-3.5-turbo',
#         messages=[
#             {"role": "system", "content": "You are a skilled Nim player."},
#             {"role": "user", "content": question}
#         ],
#         max_tokens=5,
#         temperature=0.7
#     )




# chat_llm = ChatOpenAI(temperature=0.0,
#                       streaming=True,
#                       model_name='gpt-4o-mini')



# response = chat_llm.invoke(question)
# print('gpt res:', response.choices[0].message.content)
# print('gpt res:', response.choices[0].message.content)
# print(response.content)
# print(response.content['reasoning'])
# template_string = """You are a master branding consulatant who specializes in naming brands. \
# You come up with catchy and memorable brand names.

# Take the brand description below delimited by triple backticks and use it to create the name for a brand.

# brand description: ```{brand_description}```

# then based on the description and you hot new brand name give the brand a score 1-10 for how likely it is to succeed.
# """

# def get_basic_move(agent, remaining_items):
#     prompt = f"""
#     You are {agent['name']} in a game of Nim. There are {remaining_items} items remaining in the pile.
#     You can take between 1 and {max_take} items on your turn. The goal is to win by taking all remaining items on your turn, leaving no items for your opponent.

#     Based on the current state of the game, decide how many items you will take. Provide only the integer number of items, between 1 and {max_take}, with no extra text or symbols. For example,\n
#     #Reasoning: \n
#     [reasoning content]\n
#     #Action:  \n
#     2
#     """
#     response = openai.ChatCompletion.create(
#         model=agent["model"],
#         messages=[
#             {"role": "system", "content": "You are a skilled Nim player."},
#             {"role": "user", "content": prompt}
#         ],
#         max_tokens=5,
#         temperature=0.7
#     )
#     try:
#         move = int(response.choices[0].message['content'].strip())
#         if 1 <= move <= max_take:
#             return move
#     except ValueError:
#         pass
#     return random.randint(1, max_take)










# prompt = f"""
#     You are {agent['name']} in a game of Nim. There are {remaining_items} items remaining in the pile.
#     You can take between 1 and {max_take} items on your turn. The goal is to win by taking all remaining items on your turn, leaving no items for your opponent.

#     Based on the current state of the game, decide how many items you will take. Provide only the integer number of items, between 1 and {max_take}, with no extra text or symbols. For example,\n
#     #Reasoning: \n
#     [reasoning content]\n
#     #Action:  \n
#     2
#     """

# system_message_prompt = SystemMessagePromptTemplate.from_template(
#     "당신은 {input_language}를 {output_language}로 번역하는 전문 번역가입니다."
# )

# human_message_prompt = HumanMessagePromptTemplate.from_template("{text}")

# chat_prompt = ChatPromptTemplate.from_messages(
#     [system_message_prompt, human_message_prompt]
# )

# chat_prompt.format_messages(input_language="영어", output_language="한국어", text="I love programming.")



# def play_nim_game(total_items, max_take, verbose=False):
#     current_items = total_items
#     turn = 0
#     while current_items > 0:
#         current_agent = agents[turn % 2]
#         other_agent = agents[(turn + 1) % 2]

#         if current_agent["prompting_method"] == "self_consistency":
#             move = get_consistent_move(current_agent, current_items, self_consistency_count)
#         # elif current_agent["prompting_method"] == "n_step_lookahead":
#         #     move = get_move_with_n_step_lookahead(current_agent, other_agent, current_items)
#         elif current_agent["prompting_method"] == "self_reflection":
#             move = get_move_with_reflection(current_agent, current_items)
#         elif current_agent["prompting_method"] == "debate":
#             move = get_move_with_debate(current_agent, other_agent, current_items)
#         elif current_agent["prompting_method"] == "self_play_debate":
#             move = self_play_debate(current_agent, other_agent, current_items, n_step_lookahead)
#         else:
#             # print(f"Error: Unknown prompting method '{current_agent['prompting_method']}' for {current_agent['name']}. Using basic move.")
#             move = get_basic_move(current_agent, current_items)

#         if verbose:
#             print(f"{current_agent['name']} ({current_agent['model']}) takes {move} items. Items remaining: {current_items - move}")
#         current_items -= move

#         if current_items <= 0:
#             if verbose:
#                 print(f"{current_agent['name']} ({current_agent['model']}) wins!")
#             return current_agent["name"]

#         turn += 1

# # Run the simulation
# def simulate_games(num_games, total_items, max_take):
#     win_counts = {agent["name"]: 0 for agent in agents}
#     for game_num in range(num_games):
#         print(f"\nStarting Game {game_num + 1}")
#         winner = play_nim_game(total_items, max_take, verbose=True)
#         win_counts[winner] += 1
#     print("\nGame Results:")
#     for agent in agents:
#         win_rate = (win_counts[agent["name"]] / num_games) * 100
#         print(f"{agent['name']} Win Rate: {win_rate:.2f}% ({win_counts[agent['name']]} wins out of {num_games})")

# simulate_games(num_games, total_items, max_take)