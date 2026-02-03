import os
import streamlit as st
from sentence_transformers import SentenceTransformer
import numpy as np
import torch
import pandas as pd
#pandas

# Set up page configuration and CSS.
st.set_page_config(page_title="Job Posting Search Engine", layout="centered")
st.markdown(
    """
    <style>
    .main {
        background: linear-gradient(135deg, #f8f3ff, #ffffff);
    }

    html, body, [class*="css"] {
        color: #2e003e;
        font-family: "Segoe UI", sans-serif;
    }

    .block-container {
        background-color: rgba(255, 255, 255, 0.85);
        backdrop-filter: blur(6px);
        box-shadow: 0 0 18px rgba(155, 89, 182, 0.12);
        border-radius: 12px;
        padding: 2rem;
    }

    .stApp h1 {
        color: #6a1b9a;
    }

    .stTextInput > div > input {
        background-color: #f3e5f5;
        color: #2e003e;
    }

    .stButton button {
        background-color: #7e57c2;
        color: white;
        border-radius: 6px;
    }

    .stButton button:hover {
        background-color: #5e35b1;
        transition: 0.3s;
    }

    .stAlert {
        background-color: #ede7f6 !important;
        color: #4a148c !important;
    }

    hr {
        border: none;
        border-top: 2px solid #ce93d8;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Helper: detect device.
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

st.title('Job Posting Search Engine')
device = get_device()

# Initialize session state variables.
if "selected_job" not in st.session_state:
    st.session_state.selected_job = None
# Instead of using "user_input" to preserve the search query, we use "saved_search".
if "saved_search" not in st.session_state:
    st.session_state.saved_search = ""
if "app_state" not in st.session_state:
    # "search": initial search input form,
    # "results": search results are available,
    # "similar_jobs": a job has been selected to view similar jobs.
    st.session_state.app_state = "search"

# ----- Functions for loading resources -----
@st.cache_resource
def load_fine_tuned_embeddings():
    return np.load('streamlit_app/data/fine_tuned_embeddings.npy')

@st.cache_resource
def load_default_embeddings():
    return np.load('streamlit_app/data/default_embeddings.npy')

@st.cache_resource
def load_job_postings():
    df = pd.read_parquet('streamlit_app/data/job_postings.parquet')
    df['posting'] = df['job_posting_title'] + ' @ ' + df['company']
    return df['posting'].tolist()


@st.cache_resource
def load_fine_tuned_model():
    model_path = r'fine_tuning/data/trained_models/sentence-transformers-paraphrase-MiniLM-L6-v2_triplet_2025-04-29_12-39-22'
    return SentenceTransformer(model_path, device=device)

@st.cache_resource
def load_default_model():
    return SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L6-v2', device=device)


# ----- Load Resources -----
# For demonstration, limit to the first 5000 job postings.
fine_tuned_embeddings = torch.tensor(load_fine_tuned_embeddings()[:5000], device=device)
default_embeddings = torch.tensor(load_default_embeddings()[:5000], device=device)
job_postings = load_job_postings()[:5000]
fine_tuned_model = load_fine_tuned_model()
default_model = load_default_model()


if st.session_state.app_state == "similar_jobs" and st.session_state.selected_job is not None:
    # Similar-jobs view.
    selected_index = st.session_state.selected_job
    st.header("Similar Jobs for:")
    st.write(f"**{job_postings[selected_index]}**")
    st.markdown("<hr>", unsafe_allow_html=True)
    # Compute similar jobs for both models.
    with torch.inference_mode():
        # Default model similar jobs.
        default_embedding = default_embeddings[selected_index]
        default_sim = torch.inner(default_embedding, default_embeddings)
        default_sim[selected_index] = -1  # Exclude the job itself.
        default_top_indices = torch.argsort(default_sim, descending=True)[:5]
        # Fine-tuned model similar jobs.
        finetuned_embedding = fine_tuned_embeddings[selected_index]
        finetuned_sim = torch.inner(finetuned_embedding, fine_tuned_embeddings)
        finetuned_sim[selected_index] = -1
        finetuned_top_indices = torch.argsort(finetuned_sim, descending=True)[:5]
    st.markdown(
        """
        <div class="section-spacing">
            <h3 style="margin-bottom:1rem;">Similar Jobs (Default vs. Fine-Tuned)</h3>
        </div>
        """,
        unsafe_allow_html=True,
    )
    # Display headers.
    col_rank, col_default, col_finetuned = st.columns([0.5, 4, 4])
    with col_rank:
        st.write("")  # placeholder for rank header.
    with col_default:
        st.markdown("<div class='header-container'><div>Default Model</div></div>", unsafe_allow_html=True)
    with col_finetuned:
        st.markdown("<div class='header-container'><div>Fine-Tuned Model</div></div>", unsafe_allow_html=True)
    # Show similar jobs result rows.
    for i in range(5):
        col_rank, col_default, col_finetuned = st.columns([0.5, 4, 4])
        with col_rank:
            st.markdown(f"<h4>{i+1}.</h4>", unsafe_allow_html=True)
        with col_default:
            idx = default_top_indices[i].item()
            st.write(f"**{job_postings[idx]}**")
            st.write(f"Score: {default_sim[idx]:.4f}")
        with col_finetuned:
            idx = finetuned_top_indices[i].item()
            st.write(f"**{job_postings[idx]}**")
            st.write(f"Score: {finetuned_sim[idx]:.4f}")
    st.markdown("<br><br>", unsafe_allow_html=True)
    if st.button("Back to search", key="clear_selection"):
        st.session_state.selected_job = None
        # Transition back to "results" without wiping the search query.
        st.session_state.app_state = "results"
        st.experimental_rerun()
else:
  
    user_input = st.text_input(
        "Enter a job title:",
        value=st.session_state.get("saved_search", ""),
        key="user_input"
    )
    if user_input:
        # Save the query separately so that it persists even if the widget is re-rendered.
        st.session_state.saved_search = user_input
        st.session_state.app_state = "results"
        with torch.inference_mode():
            default_query_embedding = default_model.encode(
                [user_input],
                normalize_embeddings=True,
                convert_to_tensor=True,
            )[0]
            finetuned_query_embedding = fine_tuned_model.encode(
                [user_input],
                normalize_embeddings=True,
                convert_to_tensor=True,
            )[0]
            default_sim = torch.inner(default_query_embedding, default_embeddings)
            finetuned_sim = torch.inner(finetuned_query_embedding, fine_tuned_embeddings)
            top10_default = torch.argsort(default_sim, descending=True)[:10]
            top10_finetuned = torch.argsort(finetuned_sim, descending=True)[:10]
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown(
            """
            <div class="section-spacing">
                <h3 style="margin-bottom:1rem;">Top Matches (Default vs. Fine-Tuned)</h3>
            </div>
            """,
            unsafe_allow_html=True,
        )
        # Column headers above search results.
        col_rank, col_default, col_finetuned = st.columns([0.5, 4, 4])
        with col_rank:
            st.write("")  # empty header for rank.
        with col_default:
            st.markdown("<div class='header-container'><div>Default Model</div></div>", unsafe_allow_html=True)
        with col_finetuned:
            st.markdown("<div class='header-container'><div>Fine-Tuned Model</div></div>", unsafe_allow_html=True)
        # Build search results rows.
        for i in range(len(top10_default)):
            col_rank, col_default, col_finetuned = st.columns([0.5, 4, 4])
            with col_rank:
                st.markdown(f"<h4>{i+1}.</h4>", unsafe_allow_html=True)
            with col_default:
                job_index = top10_default[i].item()
                st.write(f"**{job_postings[job_index]}**")
                st.write(f"Score: {default_sim[job_index]:.4f}")
                if st.button("Show most similar jobs", key=f"default_{job_index}"):
                    st.session_state.selected_job = job_index
                    st.session_state.app_state = "similar_jobs"
                    st.experimental_rerun()
            with col_finetuned:
                job_index = top10_finetuned[i].item()
                st.write(f"**{job_postings[job_index]}**")
                st.write(f"Score: {finetuned_sim[job_index]:.4f}")
                if st.button("Show most similar jobs", key=f"finetuned_{job_index}"):
                    st.session_state.selected_job = job_index
                    st.session_state.app_state = "similar_jobs"
                    st.experimental_rerun()
    else:
        st.info("Please enter a job title to start searching.")
