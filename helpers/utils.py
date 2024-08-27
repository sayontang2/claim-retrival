import pandas as pd
import ast
from tqdm import tqdm
import torch
import gc
import json
from copy import deepcopy as cc
gc.disable()

# -------------------------------------------------------------------------#
# PROCESS DATA
# -------------------------------------------------------------------------#
def get_data(fact_path, posts_path, post2fact_mapping_path):
    parse_col = lambda s: ast.literal_eval(s.replace('\n', '\\n')) if s else s

    # processing fact-checks data
    df_fact = pd.read_csv(fact_path).fillna('').set_index('fact_check_id')
    for col in ['claim', 'instances', 'title']:
        df_fact[col] = df_fact[col].apply(parse_col)
    df_fact['fact_check_id'] = df_fact.index.tolist()
    df_fact = df_fact.reset_index(drop=True)
    df_fact = df_fact.rename(columns={c: f'fact_{c}' for c in df_fact.columns.tolist() if not c.startswith('fact')})

    # processing posts data
    df_posts = pd.read_csv(posts_path).fillna('').set_index('post_id')
    for col in ['instances', 'ocr', 'verdicts', 'text']:
        df_posts[col] = df_posts[col].apply(parse_col)
    df_posts['post_id'] = df_posts.index.tolist()
    df_posts = df_posts.reset_index(drop=True)
    df_posts = df_posts.rename(columns={c: f'post_{c}' for c in df_posts.columns.tolist() if not c.startswith('post')})

    # processing post2fact mapping data
    df_post2fact_map = pd.read_csv(post2fact_mapping_path).reset_index(drop=True)
    return df_fact, df_posts, df_post2fact_map

def process_facts_df(df):
    df['facts_orig'] = df.fact_claim.apply(lambda x: x[0]).tolist()
    df['facts_eng'] = df.fact_claim.apply(lambda x: x[1]).tolist()
    df['facts_lang'] = df.fact_claim.apply(lambda x: x[-1][0][0]).tolist()
    return df

def process_posts_df(df):
    df['post_text'] = df.apply(lambda x: x['post_text'] if len(x['post_text']) != 0 else x['post_ocr'][0], axis = 1)
    df['post_l1'] = df.post_text.apply(lambda x: x[0]).tolist()
    df['post_l2'] = df.post_text.apply(lambda x: x[1]).tolist()
    df['l1'] = df.post_text.apply(lambda x: x[-1][0][0]).tolist()
    df['l2'] = df.post_text.apply(lambda x: x[-1][-1][0]).tolist()
    return df

# -------------------------------------------------------------------------#
# GET QUERY EMBEDDINGS
# -------------------------------------------------------------------------#
def get_sbert_embeddings(model, txt_list, batch_size=128):
    embedding_list = []

    # Initialize the progress bar
    pbar = tqdm(total=len(txt_list))

    while txt_list:
        batch = txt_list[:batch_size]
        embeddings = model.encode(batch)
        embedding_list.extend(embeddings)
        del txt_list[:batch_size]

        # Update the progress bar
        pbar.update(len(batch))

        torch.cuda.empty_cache()
        gc.collect()  
    
    # Close the progress bar
    pbar.close()

    return embedding_list

def get_openai_embedding(text_list, model, client):
   embedding_list = []

   for text in tqdm(text_list, total=len(text_list)):
      text = text.replace("\n", " ")
      embedding = client.embeddings.create(input = [text], model=model).data[0].embedding
      embedding_list.append(embedding)
   
   return embedding_list

# -------------------------------------------------------------------------#
# CREATE OPENAI BATCH INPUT
# -------------------------------------------------------------------------#
def get_openai_batch_ip(df, model, key_col, encoding_col):
    df_temp = cc(df)
    df_temp["method"] = "POST"
    df_temp["url"] = "/v1/embeddings"
    df_temp["body"] = df_temp.apply(lambda x: {"model": model, 
                                     "input": x[encoding_col].strip(), 
                                     "encoding_format": "float"}, axis=1)
    df_temp['custom_id'] = df_temp[key_col].apply(lambda x: str(x))
    df_temp = df_temp[['custom_id', 'method', 'url', 'body']]
    return df_temp