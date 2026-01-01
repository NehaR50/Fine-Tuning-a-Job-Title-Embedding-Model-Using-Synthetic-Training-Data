import sys
import os
import pandas as pd
import numpy as np
import pickle
import time
#import time
# Set path to the root of the project
#build_project_path = r"C:\Users\nehar\Downloads\Fine-Tuning-Embedding-Model-Using-Synthetic-Training-Data"
build_project_path= os.path.dirname(os.path.abspath(__file__))
synthetic_data_path = os.path.join(build_project_path, 'synthetic_data')
data_path = os.path.join(synthetic_data_path, 'data')

if synthetic_data_path not in sys.path:
    sys.path.append(synthetic_data_path)

from llm_requests import generate_prompt, get_client, make_api_call, main_stubborn
from gpt_parsing import parse_gpt_response

def main():
    print(f"Loading seed titles from: {data_path}")
    seed_titles_df = pd.read_csv(os.path.join(data_path, 'seed_titles.csv'))
    seed_titles = seed_titles_df['seed_title'].unique()
    
    # Using Groq-supported model name
    model_name = 'meta-llama/llama-4-scout-17b-16e-instruct'  # Confirmed working model
    
    print("Creating client...")
    client = get_client()
    
    # Example with a small batch of seed titles
    print("Generating example with sample titles...")
    example_seed_titles = np.random.choice(seed_titles, 5)
    example_prompt = generate_prompt(example_seed_titles)
    
    try:
        example_response = make_api_call(client, model_name, example_prompt, perturbation_std=0.1)
        parsed_output = parse_gpt_response(example_response.choices[0].message.content, 5, 5)
        
        # Check if parsing was successful
        if parsed_output is not None:
            # Print example variations
            for seed_title, response in zip(example_seed_titles, parsed_output):
                print(f'Variations of: {seed_title}:')
                print('-------------------')
                for i, variation in enumerate(response):
                    print(f'{i+1}: {variation}')
                print('\n\n')
        else:
            print("Failed to parse API response. Raw response content:")
            print(example_response.choices[0].message.content)
            print("\nSkipping example display and continuing with full generation...")
        
        # Generate variations for all seed titles
        print("Generating variations for all seed titles...")
        output_dict_path = os.path.join(data_path, 'jitter_responses.pkl')
        response_dict = main_stubborn(
            all_query_titles=seed_titles,
            client=client,
            model_name=model_name,
            output_path=output_dict_path,
            chunk_size=5,
            num_examples_per_title=5,
            giveup_after=1,
        )
        
        # Check if we got any successful responses
        if response_dict and len(response_dict) > 0:
            # Create dataframe with results
            print("Creating dataframe...")
            jitter_df = {
                'jittered_title': [],
                'seed_title': [],
            }
            
            for seed_title, jittered_titles in response_dict.items():
                for jittered_title in jittered_titles:
                    jitter_df['jittered_title'].append(jittered_title)
                    jitter_df['seed_title'].append(seed_title)
            
            jitter_df = pd.DataFrame(jitter_df)
            jitter_df = jitter_df.merge(seed_titles_df, on='seed_title', how='left')
            
            # Save results
            print("Saving results...")
            jitter_df.to_csv(os.path.join(data_path, 'jittered_titles.csv'), index=False)
            print("Done! Results saved to jittered_titles.csv")
        else:
            print("No successful responses were generated.")
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("Check if the Groq API key is configured correctly and if the model is available.")

if __name__ == "__main__":
    main()
