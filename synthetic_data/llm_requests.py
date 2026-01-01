import sys
import os
#import os
import openai
import pickle
from tqdm.autonotebook import tqdm, trange
import numpy as np
ppppop
# Set your actual project path here
#build_project_path = r"C:\Users\nehar\Downloads\Fine-Tuning-Embedding-Model-Using-Synthetic-Training-Data"
build_project_path=os.path.dirname(os.path.abspath(__file__))
synthetic_data_path = os.path.join(build_project_path, 'synthetic_data')
data_path = os.path.join(synthetic_data_path, 'data')

if synthetic_data_path not in sys.path:
    sys.path.append(synthetic_data_path)

from gpt_parsing import parse_gpt_response

# Debug print to check path
print(f"Looking for prompts at: {data_path}")

with open(os.path.join(data_path, 'initial_prompt.txt'), 'r') as f:
    base_prompt = f.read()

example_query_titles = []
with open(os.path.join(data_path, 'example_query_titles.txt'), 'r') as f:
    for line in f:
        example_query_titles.append(line.strip())

with open(os.path.join(data_path, 'example_assistant_response.txt'), 'r') as f:
    example_assistant_response = f.read()

with open(os.path.join(data_path, 'follow_up_prompt.txt'), 'r') as f:
    follow_up_prompt = f.read()

rng = np.random.default_rng()

def get_client():
    client = openai.OpenAI(
        api_key=r'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',  # Groq API key
        base_url="https://api.groq.com/openai/v1"
    )
    return client

def format_query_title_list(query_job_titles):
    output_string = ''
    for i, title in enumerate(query_job_titles):
        output_string += f'{i+1}. `{title}`\n'
    return output_string

def generate_prompt(query_job_titles, num_examples_per_title=5):
    return [
        {"role": "system", "content": "You are an expert in recruitment, staffing and HR."},
        {"role": "user", "content": f"{base_prompt.format(format_query_title_list(example_query_titles))}"},
        {"role": "assistant", "content": example_assistant_response},
        {"role": "user", "content": f"{follow_up_prompt.format(num_examples_per_title, format_query_title_list(query_job_titles))}"},
    ]

def make_api_call(client, model_name, messages, perturbation_std=0.0):
    system_message = {"role": "system", "content": "You are an expert in recruitment, staffing and HR. Always format your response as follows:\n<response>\n1. [`variation1`, `variation2`, `variation3`, `variation4`, `variation5`]\n2. [`variation1`, `variation2`, `variation3`, `variation4`, `variation5`]\n...\n</response>"}
    
    updated_messages = [system_message] + messages[1:]

    response = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",  # Groq-supported model
        messages=updated_messages,
        stop=["<query>"],
        temperature=0.7 + rng.normal(0, perturbation_std),
        max_tokens=2000
    )
    return response

def main_stubborn(all_query_titles, client, model_name, output_path=None, chunk_size=5, num_examples_per_title=5, delay=2, giveup_after=10):
    import time
    responses_dict = {}
    for i in trange(0, len(all_query_titles), chunk_size):
        attempts = 0
        current_query_titles = all_query_titles[i:i+chunk_size]
        while attempts < giveup_after:
            if attempts > 0:
                print(f'Attempt {attempts} for chunk {i // chunk_size}')
            response = make_api_call(
                client,
                model_name,
                generate_prompt(
                    query_job_titles=current_query_titles,
                    num_examples_per_title=num_examples_per_title
                ),
                perturbation_std=0.1
            )
            parsed_response = parse_gpt_response(
                gpt_output=response.choices[0].message.content,
                num_query_titles=len(current_query_titles),
                num_examples_per_query_title=num_examples_per_title,
                throw_exception_on_failure=False
            )
            if parsed_response:
                for query_title, response_list in zip(current_query_titles, parsed_response):
                    responses_dict[query_title] = response_list
                break
            elif attempts > 2:
                print('-------------------------------')
                print('Output:')
                print(response.choices[0].message.content)
            attempts += 1
            time.sleep(delay)

        if output_path:
            with open(output_path, 'wb') as f:
                pickle.dump(responses_dict, f)

    return responses_dict
