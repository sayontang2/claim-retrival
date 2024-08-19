from dataset import (create_inference_dataset, PairDatasetWithEmbeddings_for_inf, TextType, collate_inf_fn)
# from model import AggregationMethod, FeatureSet
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict
from utils import get_top_k_acc
import pandas as pd

def evaluate(model, tokenizer, device, eval_posts: pd.DataFrame, eval_facts: pd.DataFrame, eval_mapping: pd.DataFrame, 
             fact_orig_emb_pth: str, fact_eng_emb_pth: str, post_l1_emb_pth: str, post_l2_emb_pth: str,
             batch_size: int = 32, top_K: int = 5) -> Dict[str, np.ndarray]:
    """
    Evaluates the model's performance on fact retrieval tasks for a given set of posts and facts 
    using cosine similarity and top-K accuracy.

    Args:
        model: The trained model used to generate representations for posts and facts.
        tokenizer: The tokenizer used to preprocess the input text.
        device: The device (CPU/GPU) on which the evaluation is performed.
        eval_posts (pd.DataFrame): DataFrame containing the posts to be evaluated.
        eval_facts (pd.DataFrame): DataFrame containing the fact-checks to be evaluated.
        eval_mapping (pd.DataFrame): DataFrame mapping posts to their corresponding facts.
        fact_orig_emb_pth (str): Path to the original language embeddings for facts.
        fact_eng_emb_pth (str): Path to the English language embeddings for facts.
        post_l1_emb_pth (str): Path to the embeddings for posts in the first language (l1).
        post_l2_emb_pth (str): Path to the embeddings for posts in the second language (l2).
        batch_size (int): Batch size to use during evaluation.
        top_K (int): The number of top predictions to consider for accuracy calculation.

    Returns:
        Dict[str, np.ndarray]: A dictionary of top-K accuracy results for each post-fact language pair.
    """

    # Initialize data and mappings
    fact_ix2id_mapping = {ix: row['fact_check_id'] for ix, row in eval_facts.iterrows()}
    get_fact_id = lambda x: fact_ix2id_mapping[x]
    vfunc = np.vectorize(get_fact_id)
    
    eval_dict = {'post_l1': TextType.POST_L1, 'post_l2': TextType.POST_L2, 
                 'fact_orig': TextType.FACT_ORIG, 'fact_eng': TextType.FACT_ENG}
    
    eval_model_repr = {}
    temp_df = eval_mapping.copy()

    # Load and process datasets in batches
    for key, txt_type in eval_dict.items():
        text_list, small_emb_list, large_emb_list = create_inference_dataset(
            df_facts=eval_facts, df_posts=eval_posts, df_mapping=eval_mapping, 
            fact_orig_emb_pth=fact_orig_emb_pth, fact_eng_emb_pth=fact_eng_emb_pth, 
            post_l1_emb_pth=post_l1_emb_pth, post_l2_emb_pth=post_l2_emb_pth, txt_type=txt_type
        )
        
        # Create DataLoader for evaluation
        eval_dataset = PairDatasetWithEmbeddings_for_inf(
            txt_list=text_list, external_embeddings1=small_emb_list, 
            external_embeddings2=large_emb_list, tokenizer=tokenizer
        )
        
        eval_dl = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_inf_fn)
        eval_model_repr[key] = []

        # Model inference in batches
        with torch.no_grad():
            for batch in eval_dl:
                batch = {k: v.to(device) for k, v in batch.items()}
                res = model(batch).cpu().detach().numpy()
                eval_model_repr[key].extend(res)

        # Merge model representations into DataFrame
        if 'post' in key:
            df = eval_posts.copy()
            df[f'pred_{key}'] = eval_model_repr[key]
            temp_df = temp_df.merge(df[['post_id', f'pred_{key}']], on='post_id', how='left')
        if 'fact' in key:
            df = eval_facts.copy()
            df[f'pred_{key}'] = eval_model_repr[key]
            temp_df = temp_df.merge(df[['fact_check_id', f'pred_{key}']], on='fact_check_id', how='left')

    eval_results = {}
    metric_keys = [('post_l1', 'fact_eng'), ('post_l1', 'fact_orig'), ('post_l2', 'fact_eng'), ('post_l2', 'fact_orig')]

    # Efficient cosine similarity and accuracy calculation
    for post_key, fact_key in metric_keys:

        post_repr_list = np.array(temp_df[f'pred_{post_key}'].tolist())
        fact_repr_list = np.array(eval_model_repr[fact_key])
        true_fact_id_list = temp_df['fact_check_id'].tolist()

        # Calculate cosine similarity in batches
        pred = cosine_similarity(post_repr_list, fact_repr_list)

        # Get top-K accuracy
        pred_fact_ids = vfunc((-pred).argsort(axis=1)[:, :top_K])
        res = get_top_k_acc(pred_fact_ids, true_fact_id_list, top_K)
        
        eval_results[f'{post_key}_{fact_key}'] = res

    return pd.DataFrame(eval_results)