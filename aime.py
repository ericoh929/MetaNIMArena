import os
import re
import json
import time
import argparse
from collections import Counter

from tqdm import tqdm
from datasets import load_dataset
from openai import OpenAI
import google.generativeai as genai

# Argument parser
parser = argparse.ArgumentParser(description='Arguments for AIME benchmark simulation')
parser.add_argument('--agent_model', type=str, default='gpt-4o-mini', help='Agent model to use')
parser.add_argument('--agent_prompt', type=str, default='basic', help='Prompt method: basic or consistency')
args = parser.parse_args()

# Initialize client and dataset
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
dataset = load_dataset("AI-MO/aimo-validation-amc", split="train")

# get_agent_response 함수: 모델 별로 JSON 형태의 응답을 파싱하여 반환
def get_agent_response(agent, prompt, system_prompt="Output your answer as a valid JSON with keys 'reasoning' and 'answer'. The value of key 'answer' should be only integer.", temperature=0.7):
    while True:
        try:
            if agent in ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-1.0-pro"]:
                generation_config = {
                    "temperature": temperature,
                    "top_p": 0.95,
                    "top_k": 40,
                    "max_output_tokens": 8192,
                    "response_mime_type": "application/json",
                }
                model = genai.GenerativeModel(
                    model_name=agent,
                    generation_config=generation_config,
                    system_instruction=system_prompt,
                )
                chat_session = model.start_chat(history=[])
                response = chat_session.send_message(prompt)
                try:
                    return json.loads(response.text)
                except json.JSONDecodeError:
                    print("Error decoding JSON for gemini model. Retrying...")

            elif agent in ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"]:
                response = client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    model=agent,
                    response_format={"type": "json_object"},
                    temperature=temperature,
                )
                content = response.choices[0].message.content
                print('content:', content)
                parsed_content = json.loads(content)
                if parsed_content:
                    return parsed_content

            print("Error encountered. Retrying in 2 seconds...")
            time.sleep(2)  # Delay to prevent rapid retries
        except KeyboardInterrupt:
            print("Process interrupted by user.")
            return None
        except Exception as e:
            print(f"Unexpected error: {e}. Retrying...")

# Agent 모델 설정 (예: gpt-4o-mini)
agent = args.agent_model

# 기본 프롬프트 함수: 단일 응답을 받아 처리 (React)
def React(question):
    prompt = f"""Given the below question, make proper reasoning and answer.
Question: {question}
"""
    parsed_content = get_agent_response(agent, prompt)
    reasoning = parsed_content.get("reasoning")
    answer = parsed_content.get("answer")
    return reasoning, answer

# Consistency 프롬프트 함수: 여러 번 응답을 받아 다수결로 최종 답 산출
num_responses = 10
def Consistency(question):
    prompt = f"""Given the below question, make proper reasoning and answer for the question given the format.
Question: {question}
"""
    moves = []
    reasoning = None
    for _ in range(num_responses):
        parsed_content = get_agent_response(agent, prompt)
        reasoning = parsed_content.get("reasoning")
        answer = parsed_content.get("answer")
        try:
            move = int(answer)
        except Exception:
            move = answer  # 혹은 에러 처리를 추가
        moves.append(move)
    most_common_move = Counter(moves).most_common(1)[0][0]
    return reasoning, most_common_move


num_refine = 3
def Reflection(question):
    prompt_initial =f"""Given the below question, make proper reasoning and answer for the question given the format.
Question: {question}
""" 

    parsed_content = get_agent_response(agent, prompt_initial)

    initial_reasoning = parsed_content.get("reasoning")
    answer = parsed_content.get("answer")

    try:
        initial_move = int(answer)
    except Exception:
        initial_move = answer  # 혹은 에러 처리를 추가

    for k in range(num_refine):

        feedback_prompt = f"""
        Given the below question, make proper reasoning and answer for the question given the format.
Question: {question}
#First trial's reasoning and action:\nYou initially answered as: {initial_move} at first trial by the reason: '{initial_reasoning}'.
"""

        parsed_content = get_agent_response(agent, feedback_prompt, system_prompt="Output your feedback as a valid JSON with key 'feedback'.")
        feedback = parsed_content.get("feedback")

        refine_prompt = f"""
        Given the below question, make proper reasoning and answer for the question given the format.
Question: {question}
You initially answered as {initial_move} at first trial by the reason: '{initial_reasoning}'.
You recieved feedback on your answer and reasoning: {feedback}
Based on the feedback, refine your reasoning and answer for the Question.
"""

        parsed_content = get_agent_response(agent, refine_prompt)
        refined_reasoning = parsed_content.get("reasoning")
        refined_answer = parsed_content.get("answer")

        if initial_move == int(refined_answer):
            return refined_reasoning, refined_answer
        else:
            initial_move = refined_answer
            initial_reasoning = refined_reasoning

    return refined_reasoning, refined_answer

debate_rounds = 3
def Debate(question):
    initial_moves = {}
    initial_reasonings = {}
    i = 0
    for _ in ['1', '2']:
        prompt = f"""Given the below question, make proper reasoning and answer for the question given the format.
Question: {question}
""" 
        
        parsed_content = get_agent_response(agent, prompt)

        initial_reasoning = parsed_content.get("reasoning")
        initial_answer = parsed_content.get("answer")
        if i == 0:
            initial_moves['agent1'] = initial_answer
            initial_reasonings['agent1'] = initial_reasoning
        if i == 1:
            initial_moves['agent2'] = initial_answer
            initial_reasonings['agent2'] = initial_reasoning
        i += 1

    for _ in range(debate_rounds):
        i = 0
        for k in ['1', '2']:
            
            if i == 0:
                prompt = f"""Given the below question, make proper reasoning and answer for the question given the format.
Question: {question}

You initially answered as {initial_moves['agent1']} at first trial by the reason: '{initial_reasonings['agent1']}'.
Other agent argues that you have to answer as: {initial_moves['agent2']} by the reason: {initial_reasonings['agent2']}.
Considering the other's opinion, refine or confirm your move.
"""
            if i == 1:
                prompt = f"""Given the below question, make proper reasoning and answer for the question given the format.
Question: {question}

You initially answered as {initial_moves['agent2']} at first trial by the reason: '{initial_reasonings['agent2']}'.
Other agent argues that you have to answer as: {initial_moves['agent1']} by the reason: {initial_reasonings['agent1']}.
Considering the other's opinion, refine or confirm your move.
"""

            parsed_content = get_agent_response(agent, prompt, system_prompt="You are debating the correct answer. Output your answer as a valid JSON with keys 'reasoning' and 'answer'. The value of key 'answer' should be only integer.")

            initial_reasoning = parsed_content.get("reasoning")
            initial_answer = parsed_content.get("answer")
            if i == 0:
                a0_action = initial_answer
                a0_reasoning = initial_reasoning
                
                # print('debate round:, ', t, 'my action: ', initial_action)
                # print('debate round:, ', t, 'my reasoning: ', initial_reasoning)    
            if i == 1:
                a1_action = initial_answer
                a1_reasoning = initial_reasoning
                
                # print('debate round:, ', t, 'others action: ', initial_action)
                # print('debate round:, ', t, 'others reasoning: ', initial_reasoning)   
            i += 1
        initial_moves['agent1'] = a0_action
        initial_reasonings['agent1'] = a0_reasoning
        initial_moves['agent2'] = a1_action
        initial_reasonings['agent2'] = a1_reasoning
            
        if len(set(initial_moves.values())) == 1:
            return initial_reasoning, initial_answer

    return initial_reasoning, Counter(initial_moves.values()).most_common(1)[0][0]  # Use most common if no consensus

def Dreamad(question):
    initial_moves = {}
    initial_reasonings = {}
    i = 0
    for _ in ['1']:
        prompt = f"""Question: {question}""" 

        math_prompt = f"""
You are a helpful math strategist. Below is a math problem description.

Task:
1. Identify the type of problem (e.g., geometry, algebra, probability, etc.)
2. Summarize all relevant information and constraints in the problem.
3. Suggest a possible approach to solve it.

Remember:
- Use the exact keys: "problem_type", "given_conditions", "solution_approach"
- Do NOT include any extra text besides the JSON

**Question Description:**
{prompt}
"""
    
        parsed_content = get_agent_response(agent, math_prompt, system_prompt="You are a good math strategist. Output your answer as a valid JSON with keys 'problem_type', 'given_conditions', and 'solution_approach'.",temperature=0.1)

        problem_type = parsed_content.get("problem_type")
        given_conditions = parsed_content.get("given_conditions")
        solution_approach = parsed_content.get("solution_approach")


    for _ in [1, 2]:

        final_prompt = f"""
You are an expert math problem rewriter. 
Re-express the question in a clearer, more structured way—maintaining its original intent,
but filling in missing details or clarifying ambiguous points if needed.

Constraints:
- Do NOT change the fundamental meaning of the problem.
- If the original problem is already clear, try adding helpful clarifications or step-by-step structure.


Problem Type: {problem_type}  
Given Condition: {given_conditions}  
Solution Approach: {solution_approach}

**Initial Question Description:**
{prompt}
"""
        # print("Strategy Prompt: ", strategy_prompt)
        parsed_content = get_agent_response(agent, final_prompt, system_prompt="Output your answer as a valid JSON with keys 'refined_question'.", temperature=0.7)


        optimized_prompt = parsed_content.get("refined_question")

        one_new_prompt = f"""Given the below question, make proper reasoning and answer for the question given the format.
Question: {optimized_prompt}
        """
        # print("One New Prompt: ", one_new_prompt)
        one_parsed_content = get_agent_response(agent, one_new_prompt)
        # print(4)
        one_reasoning = one_parsed_content.get("reasoning")
        one_answer = one_parsed_content.get("answer")

        if i == 0:
            initial_moves['agent1'] = one_answer
            initial_reasonings['agent1'] = one_reasoning
            one_prompt = optimized_prompt
        if i == 1:
            initial_moves['agent2'] = one_answer
            initial_reasonings['agent2'] = one_reasoning
            two_prompt = optimized_prompt

        i += 1


    for _ in range(debate_rounds):
        i = 0
        for _ in [1, 2]:
            
            if i == 0:
                prompt = f"""Given the below question, make proper reasoning and answer for the question given the format.
Question: {one_prompt}

You initially answered as {initial_moves['agent1']} at first trial by the reason: '{initial_reasonings['agent1']}'.
Other agent argues that you have to answer as: {initial_moves['agent2']} by the reason: {initial_reasonings['agent2']}.
Considering the other's opinion, refine or confirm your move.
"""
            if i == 1:
                prompt = f"""Given the below question, make proper reasoning and answer for the question given the format.
Question: {two_prompt}

You initially answered as {initial_moves['agent2']} at first trial by the reason: '{initial_reasonings['agent2']}'.
Other agent argues that you have to answer as: {initial_moves['agent1']} by the reason: {initial_reasonings['agent1']}.
Considering the other's opinion, refine or confirm your move.
"""

            parsed_content = get_agent_response(agent, prompt, system_prompt="You are debating the correct answer. Output your answer as a valid JSON with keys 'reasoning' and 'answer'. The value of key 'answer' should be only integer.")

            initial_reasoning = parsed_content.get("reasoning")
            initial_answer = parsed_content.get("answer")
            if i == 0:
                a0_action = initial_answer
                a0_reasoning = initial_reasoning
                
                # print('debate round:, ', t, 'my action: ', initial_action)
                # print('debate round:, ', t, 'my reasoning: ', initial_reasoning)    
            if i == 1:
                a1_action = initial_answer
                a1_reasoning = initial_reasoning
                
                # print('debate round:, ', t, 'others action: ', initial_action)
                # print('debate round:, ', t, 'others reasoning: ', initial_reasoning)   
            i += 1
        initial_moves['agent1'] = a0_action
        initial_reasonings['agent1'] = a0_reasoning
        initial_moves['agent2'] = a1_action
        initial_reasonings['agent2'] = a1_reasoning
            
        if len(set(initial_moves.values())) == 1:
            return initial_reasoning, initial_answer

    return initial_reasoning, Counter(initial_moves.values()).most_common(1)[0][0]  # Use most common if no consensus

def simulate(prompt_method):
    correct = 0
    total = 0
    num = 0
    for item in tqdm(dataset):
        # num += 1
        # if num < 25:
        #     continue
        # 데이터셋의 문제와 정답 키 (필요에 따라 키 이름을 수정)
        question = item.get("problem")
        true_answer = int(item.get("answer"))
        if question is None or true_answer is None:
            continue

        reasoning, answer = prompt_method(question)

        # 단순 비교: 문자열 혹은 정수 비교 (필요시 전처리 추가)
        if answer == true_answer:
            correct += 1
        total += 1

        # 각 문제의 결과 출력
        print("문제:", question)
        print("모델 답변:", answer)
        print("실제 정답:", true_answer)
        print("-" * 50)

    accuracy = correct / total if total > 0 else 0
    print(f"\n전체 문제 수: {total}")
    print(f"정답 수: {correct}")
    print(f"Accuracy: {accuracy * 100:.2f}%")

# 메인 함수: argument로 전달된 프롬프트 방법에 따라 시뮬레이션 실행
def main():
    prompt_method_name = args.agent_prompt.lower()
    if prompt_method_name in ["consistency"]:
        prompt_method = Consistency
    elif prompt_method_name in ["reflection"]:
        prompt_method = Reflection
    elif prompt_method_name in ["debate"]:
        prompt_method = Debate
    elif prompt_method_name in ["dreamad"]:
        prompt_method = Dreamad
    else:  # 기본은 React 방식 (basic)
        prompt_method = React

    simulate(prompt_method)

if __name__ == "__main__":
    main()

# import os
# from openai import OpenAI
# import google.generativeai as genai

# genai.configure(api_key='AIzaSyCYkix3fQio-WpUus7ziwNGhSk6qZp7LJs')

# client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"),)
# import re
# from collections import Counter
# import argparse

# from datasets import load_dataset
# from transformers import pipeline
# from tqdm import tqdm
# import time
# import json

# parser = argparse.ArgumentParser(description='arguments for training')

# parser.add_argument('--agent_model',     type=str,   default=None, help='model')
# parser.add_argument('--agent_prompt',     type=str,   default='basic', help='prompt_method')

# args = parser.parse_args()

# dataset = load_dataset("Maxwell-Jia/AIME_2024", split="train")

# def get_agent_response(agent, prompt, system_prompt="Output your answer as a valid JSON with keys 'reasoning' and 'answer'.", temperature = 0.7):
#     while True:
#         try:
#             if agent in ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-1.0-pro"]:
#                 generation_config = {
#                     "temperature": temperature,
#                     "top_p": 0.95,
#                     "top_k": 40,
#                     "max_output_tokens": 8192,
#                     "response_mime_type": "application/json",
#                 }
#                 model = genai.GenerativeModel(
#                     model_name=agent,
#                     generation_config=generation_config,
#                     system_instruction=system_prompt,
#                 )
#                 chat_session = model.start_chat(history=[])
#                 response = chat_session.send_message(prompt)
#                 try:
#                     return json.loads(response.text)
#                 except json.JSONDecodeError:
#                     print("Error decoding JSON for gemini model. Retrying...")

#             elif agent in ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"]:
#                 response = client.chat.completions.create(
#                     messages=[
#                         {"role": "system", "content": system_prompt},
#                         {"role": "user", "content": prompt},
#                     ],
#                     model=agent,
#                     response_format={"type": "json_object"},
#                     temperature=temperature,
#                 )
#                 content = response.choices[0].message.content
#                 print('content: ', content)
#                 # match = re.search(r"```json\s*(\{.*?\})\s*```", content, re.DOTALL)
#                 # if match:
#                 #     json_str = match.group(1)
#                 # else:
#                 #     # 만약 ```json``` 태그가 없다면, 첫 번째 { } 블록을 추출합니다.
#                 #     json_str = re.search(r'\{.*?\}', content, re.DOTALL).group(0)

#                 # # 불필요한 non-breaking space를 제거하고 좌우 공백을 제거합니다.
#                 # json_str = json_str.replace('\xa0', '').strip()

#                 # # JSON 문자열을 파이썬 딕셔너리로 파싱합니다.
#                 # parsed_content = json.loads(json_str)
#                 parsed_content = json.loads(content)

#                 # parsed_content = json.loads(re.search(r'\{.*?\}', content, re.DOTALL).group(0).replace('\xa0', '').strip())
#                 if parsed_content:
#                     return parsed_content

#             print("Error encountered. Retrying in 2 seconds...")
#             time.sleep(2)  # Delay to prevent rapid retries
#         except KeyboardInterrupt:
#             print("Process interrupted by user.")
#             return None
#         except Exception as e:
#             print(f"Unexpected error: {e}. Retrying...")

# agent = 'gpt-4o-mini'


# def React(question):

#     prompt = f"""Given the below question, make proper reasoning and answer.\n
#     Question: {question}\n
#     """

#     parsed_content = get_agent_response(agent, prompt)

#     reasoning = parsed_content.get("reasoning")
#     answer = parsed_content.get("answer")

#     return reasoning, answer

# num_responses = 10
# def Consistency(question):

#     prompt = f"""Given the below question, make proper reasoning and answer for the question given the format.\n
#     Question: {question}\n

#     ### Format Response as:
#     {{
#     "reasoning": "string", // This is the reasons for the answer.
#     "answer": integer // This is an answer based on the reasoning.
#     }}
#     """

#     moves = []

#     for _ in range(num_responses):
#         parsed_content = get_agent_response(agent, prompt)

#         reasoning = parsed_content.get("reasoning")
#         answer = parsed_content.get("answer")
#         move = int(answer)
#         moves.append(move)
    
#     most_common_move = Counter(moves).most_common(1)[0][0]

#     return reasoning, most_common_move

# correct = 0
# total = 0

# # 데이터셋의 각 항목(item)은 문제와 정답 정보를 포함한다고 가정합니다.
# # 실제 키 이름이 다를 경우 "question"이나 "answer" 부분을 수정하세요.
# def simulate():
#     for item in tqdm(dataset):
#         # 예시로, 문제 텍스트는 "question" 또는 "problem" 키에, 정답은 "answer" 키에 있다고 가정합니다.
#         question = item.get("Problem")
#         true_answer = item.get("Answer")

#         _, answer = React(question)

#         # 간단 비교: 좌우 공백 제거 후 정답 문자열이 동일하면 정답으로 처리합니다.
#         if answer == true_answer:
#             correct += 1
#         total += 1

#         # 각 문제의 결과를 출력 (원한다면 주석 처리 가능)
#         print("문제:", question)
#         print("모델 답변:", answer)
#         print("실제 정답:", true_answer)
#         print("-" * 50)

#     # Accuracy 계산 및 출력
#     accuracy = correct / total if total > 0 else 0
#     print(f"\n전체 문제 수: {total}")
#     print(f"정답 수: {correct}")
#     print(f"Accuracy: {accuracy * 100:.2f}%")