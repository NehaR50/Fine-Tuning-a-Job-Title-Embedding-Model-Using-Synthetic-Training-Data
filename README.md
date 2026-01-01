# Fine-Tuned Job Title Embedding Model using Synthetic Training Data

This is an interactive Streamlit web application that allows users to search for job titles and discover similar job postings using both a default sentence embedding model and a fine-tuned model trained with synthetic data. This project demonstrates a domain-specific sentence embedding model for comparing job titles more effectively than general-purpose models. It uses synthetic variations of job titles (jittered titles) for fine-tuning, enabling enhanced semantic similarity detection in employment-related applications. A visual interface using Streamlit and clustering visualizations using t-SNE are also provided. It combines contrastive learning with LLM-driven data generation, offering a scalable and domain-specific alternative to traditional string matching or off-the shelf embeddings.


## What This Project Does

 📚 Generates synthetic variations of job titles using LLMs.
      
 🎯 Fine-tunes a lightweight transformer model using triplet loss.
      
  🔍 Builds a search engine that compares job title similarity.
      
 📊 Visualizes how embeddings improve after fine-tuning.
      
 🧠 Benchmarks popular LLM APIs for the same task.


## Features

  🔍 Search for job titles and get top matches.
      
 🤖 Compare results from:
      SentenceTransformer (default)
      Fine-tuned model (trained using triplet loss)
          
 📊 View similar job titles based on your selected result.
      
 ⚡ Powered by Sentence Transformers, PyTorch, and Streamlit.



## Project Overview
### Core Components

| Component         | Description                                                                    |
| ----------------- | ------------------------------------------------------------------------------ |
| `synthetic_data/` | LLM-based generation of jittered job title variants                            |
| `fine_tuning/`    | Fine-tuning pipeline using Triplet Loss with `SentenceTransformerTrainer`      |
| `streamlit_app/`  | Interactive search engine that compares base and fine-tuned similarity results |
| `visualization`   | t-SNE plots to evaluate semantic clustering of embeddings                      |


## Dataset Construction

We crafted a contrastive dataset consisting of:

   seed_title: Canonical job title (e.g., "Software Engineer")
    
   jittered_title: Variant generated using LLMs (e.g., "Software Eng II (Fullstack)")
    
   onet_code: ONET job classification code for stratified splitting

**Synthetic variations were generated using**:

   Initial Prompting (initial_prompt.txt)
    
   Follow-Up Prompting (follow_up_prompt.txt)
    
   Intermediate Jitter Strategies (intermediate_jitter_prompt.txt)
    
   Assistant responses logged in .txt and .pkl formats.

## Model Training Summary
We fine-tuned the paraphrase-MiniLM-L6-v2 model using triplet loss to improve job title similarity embeddings.

  ### Setup
  
  **Model**: paraphrase-MiniLM-L6-v2 (384d)
  
  **Loss**: TripletLoss with margin 0.3
  
  **Structure**: Anchor = jittered title, Positive = seed title, Negative = random seed title
  
  **Dataset**: Split into train/val/test (train_ds.csv, val_ds.csv, test_ds.csv)
  
  **Triplet Generation**: Dynamic via IterableDataset

  ### Training Config
  
  **Epochs**: 5
  
  **Batch Size**: 64
  
  **Eval per Epoch**: 4 times
  
  **Optimizer**: AdamW with warm-up
  
  **Checkpoints**: Saved to fine_tuning/data/trained_models/
  
  **Final Model**: Saved to streamlit_app/data/fine_tuned_model/

 ## Embedding Visualization
 We used t-SNE to reduce high-dimensional embeddings (768D) to 2D.

 ## Outcome:

     Fine-tuned model forms tight, distinct clusters
      
     Default model shows more diffuse, overlapping embeddings
      
  ![image alt](https://github.com/NehaR50/Fine-Tuning-a-Job-Title-Embedding-Model-Using-Synthetic-Training-Data/blob/e09fc5b832c9097753086763dd918f329281bdca/fine_tuning/embedding_visualization.png)



## Deployed on Streamlit Cloud - LIVE DEMO
**Hosted on Streamlit Cloud**. 👉 Try it here: https://nehar50-fine-tuning-a-job-title-embeddi-streamlit-appapp-us2ntv.streamlit.app/



## Results
**Improved Clustering**: The fine-tuned model demonstrates tighter clustering of semantically similar job titles.

**Enhanced Similarity Scores**: Higher accuracy in measuring job title similarities compared to baseline models.

**Interactive Exploration**: The Streamlit app allows users to compare similarity scores between the default and fine-tuned models in real-time.

