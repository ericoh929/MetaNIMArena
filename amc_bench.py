import os
import re
import json
import time
import argparse
from collections import Counter
import contextlib
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
dataset_name = "AI-MO/aimo-validation-amc" 
data = 'amc'

dataset = load_dataset(dataset_name, split="train")

prompt_type = args.agent_prompt  # argparse에서 받아온 prompt 타입
agent_type = args.agent_model  # argparse에서 받아온 agent 모델

# 옵션에 따라 파일명을 동적으로 생성 (필요시 타임스탬프 추가 가능)
timestamp = time.strftime("%Y%m%d_%H%M%S")

filename = f"results/{data}_{prompt_type}_{agent_type}_{timestamp}.txt"

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

def Dreamad_new(question):
    initial_moves = {}
    initial_reasonings = {}
    i = 0
    for _ in ['1']:
        prompt = f"""Question: {question}""" 

        math_prompt = f"""
You are a helpful math strategist. Below is a math problem description.

**Question Description:**
{prompt}

Task:
1. Identify the type of problem (e.g., geometry, algebra, probability, etc.)
2. Summarize all relevant information and constraints in the problem.
3. Suggest a good strategy to solve it.
"""
    
        parsed_content = get_agent_response(agent, math_prompt, system_prompt="You are a good math strategist. Output your answer as a valid JSON with keys 'problem_type', 'given_conditions', and 'solution_strategy'.",temperature=0.1)

        problem_type = parsed_content.get("problem_type")
        given_conditions = parsed_content.get("given_conditions")
        solution_approach = parsed_content.get("solution_strategy")


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
Solution Strategy: {solution_approach}

**Initial Question Description:**
{prompt}
"""
        # print("Strategy Prompt: ", strategy_prompt)
        parsed_content = get_agent_response(agent, final_prompt, system_prompt="Output your answer as a valid JSON with keys 'refined_question'.", temperature=0.7)


        optimized_prompt = parsed_content.get("refined_question")

        one_new_prompt = f"""Given the below question, make proper reasoning and answer for the question given the format.

Question: {optimized_prompt}

Problem Type: {problem_type}  
Given Condition: {given_conditions}  
Solution Strategy: {solution_approach}
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

def Dreamad_short(question):
    initial_moves = {}
    initial_reasonings = {}
    i = 0
    for _ in ['1']:
        prompt = f"""Question: {question}""" 

        math_prompt = f"""
You are a helpful math strategist. Extract the key information from the problem below.

**Problem Description:**
{prompt}

Task:
1. Identify the type of problem (e.g., geometry, algebra, probability, etc.)
2. Summarize all relevant information and constraints in the problem.
3. Suggest a good strategy to solve it.
"""
    
        parsed_content = get_agent_response(agent, math_prompt, system_prompt="You are a good math strategist. Output your answer as a valid JSON with keys 'problem_type', 'given_conditions', and 'solution_strategy'.",temperature=0.1)

        problem_type = parsed_content.get("problem_type")
        given_conditions = parsed_content.get("given_conditions")
        solution_approach = parsed_content.get("solution_strategy")


    for _ in [1, 2]:
        final_prompt = f"""Here is the problem along with an overview of the Extracted summary and strategy:
==================================================
Question: {prompt}

[Extracted Summary and Strategy]
Problem Type: {problem_type}  
Given Condition: {given_conditions}  
Solution Strategy: {solution_approach}
==================================================

Task:
Based on this information, refine the question in a clearer way—maintaining its original intent.

Constraints:
- Do NOT change the fundamental meaning of the problem.
- If the original problem is already clear, try adding helpful clarifications or step-by-step structure.
"""
        parsed_content = get_agent_response(agent, final_prompt, system_prompt="You are an expert math problem rewriter. Output your answer as a valid JSON with keys 'refined_question'.", temperature=0.7)

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
                
            if i == 1:
                a1_action = initial_answer
                a1_reasoning = initial_reasoning
                
            i += 1
        initial_moves['agent1'] = a0_action
        initial_reasonings['agent1'] = a0_reasoning
        initial_moves['agent2'] = a1_action
        initial_reasonings['agent2'] = a1_reasoning
            
        if len(set(initial_moves.values())) == 1:
            return initial_reasoning, initial_answer

    return initial_reasoning, Counter(initial_moves.values()).most_common(1)[0][0]  # Use most common if no consensus

def Dreamad_one(question):
    initial_moves = {}
    initial_reasonings = {}
    i = 0
    for _ in ['1']:
        prompt = f"""Question: {question}""" 

        math_prompt = f"""
Extract the key information from the problem below.

**Problem Description:**
{prompt}

Task:
1. Identify the type of problem (e.g., geometry, algebra, probability, etc.)
2. Suggest a good strategy to solve it.
"""
    
        parsed_content = get_agent_response(agent, math_prompt, system_prompt="You are a good math strategist. Output your answer as a valid JSON with keys 'problem_type', and 'solution_strategy'.",temperature=0.1)

        problem_type = parsed_content.get("problem_type")
        # given_conditions = parsed_content.get("given_conditions")
        solution_approach = parsed_content.get("solution_strategy")


    for _ in [1, 2]:
        final_prompt = f"""Here is the problem along with an overview of the Extracted summary and strategy:

Question: {prompt}

[Extracted Summary and Strategy]
Problem Type: {problem_type}  
Solution Strategy: {solution_approach}

Task:
Based on this information, refine the question in a clearer way—maintaining its original intent.

Constraints:
- Do NOT change the fundamental meaning of the problem.
- If the original problem is already clear, try adding helpful clarifications or step-by-step structure.
"""
        parsed_content = get_agent_response(agent, final_prompt, system_prompt="You are an expert math problem rewriter. Output your answer as a valid JSON with keys 'trimmed_question'.", temperature=0.7)

        optimized_prompt = parsed_content.get("trimmed_question")

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
                
            if i == 1:
                a1_action = initial_answer
                a1_reasoning = initial_reasoning
                
            i += 1
        initial_moves['agent1'] = a0_action
        initial_reasonings['agent1'] = a0_reasoning
        initial_moves['agent2'] = a1_action
        initial_reasonings['agent2'] = a1_reasoning
            
        if len(set(initial_moves.values())) == 1:
            return initial_reasoning, initial_answer

    return initial_reasoning, Counter(initial_moves.values()).most_common(1)[0][0]  # Use most common if no consensus


def Dreamad_fill(question):
    initial_moves = {}
    initial_reasonings = {}
    i = 0
    for _ in ['1']:
        prompt = f"""Question: {question}""" 

        math_prompt = f"""
Extract the key information from the problem below.

{prompt}

Task:
1. Identify the type of problem (e.g., geometry, algebra, probability, etc.)
2. Summarize all relevant information and constraints in the problem.
3. Suggest a good strategy to solve it.
"""
    
        parsed_content = get_agent_response(agent, math_prompt, system_prompt="You are a good math strategist. Output your answer as a valid JSON with keys 'problem_type', 'given_conditions', and 'solution_strategy'.",temperature=0.1)

        problem_type = parsed_content.get("problem_type")
        given_conditions = parsed_content.get("given_conditions")
        solution_approach = parsed_content.get("solution_strategy")


    for _ in [1, 2]:
        final_prompt = f"""Here is the problem along with an overview of the Extracted summary and strategy:

{prompt}

[Extracted Summary and Strategy]
Problem Type: {problem_type}  
Given Condition: {given_conditions}  
Solution Strategy: {solution_approach}

Task:
Based on this information, fill in missing details or clarify ambiguous points if needed maintaining its original intent.

Constraints:
- Do NOT change the fundamental meaning of the problem.
- If the original problem is already clear, try adding helpful clarifications or step-by-step structure.
"""
        parsed_content = get_agent_response(agent, final_prompt, system_prompt="You are an math problem expert. Output your answer as a valid JSON with keys 'trimmed_question'.", temperature=0.7)

        optimized_prompt = parsed_content.get("trimmed_question")

        one_new_prompt = f"""Given the below question, make proper reasoning and answer for the question given the format.

Problem Type: {problem_type}  
Given Condition: {given_conditions}  
Solution Strategy: {solution_approach}

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
                
            if i == 1:
                a1_action = initial_answer
                a1_reasoning = initial_reasoning
                
            i += 1
        initial_moves['agent1'] = a0_action
        initial_reasonings['agent1'] = a0_reasoning
        initial_moves['agent2'] = a1_action
        initial_reasonings['agent2'] = a1_reasoning
            
        if len(set(initial_moves.values())) == 1:
            return initial_reasoning, initial_answer

    return initial_reasoning, Counter(initial_moves.values()).most_common(1)[0][0]  # Use most common if no consensus

def Dreamad_good(question):
    initial_moves = {}
    initial_reasonings = {}
    i = 0
    for _ in ['1']:
        prompt = f"""Question: {question}""" 

        math_prompt = f"""
You are a helpful math strategist. Extract the key information from the problem below and redefine the problem.

**Problem Description:**
{prompt}

Task:
1. Identify the type of problem (e.g., geometry, algebra, probability, etc.)
2. Summarize all relevant information and constraints in the problem.
3. Suggest a good strategy to solve it.
"""
    
        parsed_content = get_agent_response(agent, math_prompt, system_prompt="You are a good math strategist. Output your answer as a valid JSON with keys 'problem_type', 'given_conditions', and 'solution_strategy'.",temperature=0.1)

        problem_type = parsed_content.get("problem_type")
        given_conditions = parsed_content.get("given_conditions")
        solution_approach = parsed_content.get("solution_strategy")

        strategy_prompt = f"""Below is the key information extracted from the problem. Re-express the question in a clearer, more structured way—maintaining its original intent, but filling in missing details or clarifying ambiguous points if needed.
Problem Type: {problem_type}  
Given Condition: {given_conditions}  
Solution Strategy: {solution_approach}

Based on the information above, summarize the main approach and strategy for how to solve this problem.

Question: {prompt}
"""
        
        parsed_content = get_agent_response(agent, strategy_prompt, system_prompt="Output your answer as a valid JSON with keys 'approach_outline', 'key_equations', and 'common_pitfalls'.", temperature=0.1)

        approach_outline = parsed_content.get("approach_outline")
        key_equations = parsed_content.get("key_equations")
        common_pitfalls = parsed_content.get("common_pitfalls")

    for _ in [1, 2]:

        

        final_prompt = f"""You are an expert math problem rewriter. Here is the problem along with an overview of the solution strategy:
==================================================
Question: {prompt}

[Extracted Summary]
Problem Type: {problem_type}  
Given Condition: {given_conditions}  
Solution Strategy: {solution_approach}

[Solution Strategy Overview]
- Approach Outline: {approach_outline}
- Key Equations: {key_equations}
- Common Pitfalls: {common_pitfalls}
==================================================

Task:
Based on this information, refine the question in a clearer, more structured way—maintaining its original intent. Refine the question to be presented its evidence and reasoning gradually, leading it step by step to the conclusion.
"""
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
        # if num < 81:
        #     continue
        # 데이터셋의 문제와 정답 키 (필요에 따라 키 이름을 수정)
        question = item.get("problem")
        true_answer = int(item.get("answer"))
        if question is None or true_answer is None:
            continue

        reasoning, answer = prompt_method(question)

        # 단순 비교: 문자열 혹은 정수 비교 (필요시 전처리 추가)
        total += 1
        if type(answer) == str:
            continue
        if answer == true_answer:
            correct += 1
        

        # 각 문제의 결과 출력
        print("문제 {total}: ", question)
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
    elif prompt_method_name in ["dreamad_new"]:
        prompt_method = Dreamad_new
    elif prompt_method_name in ["dreamad_good"]:
        prompt_method = Dreamad_good
    elif prompt_method_name in ["dreamad_fill"]:
        prompt_method = Dreamad_fill
    elif prompt_method_name in ["dreamad_short"]:
        prompt_method = Dreamad_short
    elif prompt_method_name in ["dreamad_one"]:
        prompt_method = Dreamad_one
    elif prompt_method_name in ["react"]:
        prompt_method = React
    else:  # 기본은 React 방식 (basic)
        print("Invalid prompt method. Designate a prompt method.")

    simulate(prompt_method)

if __name__ == "__main__":
    with open(filename, 'w', encoding='utf-8') as f:
        with contextlib.redirect_stdout(f):
            main()
 # type: ignore