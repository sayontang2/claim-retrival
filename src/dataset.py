import pandas as pd
import ast
from tqdm import tqdm
import torch
import gc
from copy import deepcopy as cc
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from typing import List, Tuple, Any
from enum import Enum
gc.disable()
import pdb

# -------------------------------------------------------------------------#
# DATA PROCESSING FUNCTIONS
# -------------------------------------------------------------------------#

def get_raw_data_df(fact_path: str, posts_path: str, post2fact_mapping_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Loads and processes fact-checks, posts, and post-to-fact mapping data from CSV files, returning 
    the processed data as pandas DataFrames.

    Args:
        fact_path (str): Path to the CSV file containing fact-checks data.
        posts_path (str): Path to the CSV file containing posts data.
        post2fact_mapping_path (str): Path to the CSV file containing post-to-fact mapping data.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Three DataFrames representing fact-check data, 
        posts data, and post-to-fact mapping data respectively.
    """
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
    gc.collect()
    return df_fact, df_posts, df_post2fact_map

def process_facts_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Processes a DataFrame containing fact-check data to extract and add columns for the original claims, 
    English claims, and the claim language.

    Args:
        df (pd.DataFrame): A DataFrame containing fact-check data with a 'fact_claim' column.

    Returns:
        pd.DataFrame: The modified DataFrame with added columns for original claims, English claims, 
        and claim language.
    """
    df['facts_orig'] = df.fact_claim.apply(lambda x: x[0]).tolist()
    df['facts_eng'] = df.fact_claim.apply(lambda x: x[1]).tolist()
    df['facts_lang'] = df.fact_claim.apply(lambda x: x[-1][0][0]).tolist()
    return df

def process_posts_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Processes a DataFrame containing posts data to extract and add columns for the main text of the post,
    and linguistic attributes such as first and second-level language features.

    Args:
        df (pd.DataFrame): A DataFrame containing posts data with columns such as 'post_text' and 'post_ocr'.

    Returns:
        pd.DataFrame: The modified DataFrame with added columns for post text and linguistic features.
    """
    df['post_text'] = df.apply(lambda x: x['post_text'] if len(x['post_text']) != 0 else x['post_ocr'][0], axis = 1)
    df['post_l1'] = df.post_text.apply(lambda x: x[0]).tolist()
    df['post_l2'] = df.post_text.apply(lambda x: x[1]).tolist()
    df['l1'] = df.post_text.apply(lambda x: x[-1][0][0]).tolist()
    df['l2'] = df.post_text.apply(lambda x: x[-1][-1][0]).tolist()
    return df

# -------------------------------------------------------------------------#
# EMBEDDING FUNCTIONS
# -------------------------------------------------------------------------#

def get_sbert_embeddings(model: Any, txt_list: List[str], batch_size: int = 128) -> List[List[float]]:
    """
    Generates embeddings for a list of input texts using a Sentence-BERT model, processed in batches.

    Args:
        model (Any): The Sentence-BERT model instance used to generate the embeddings.
        txt_list (List[str]): A list of input texts to generate embeddings for.
        batch_size (int, optional): The number of texts to process per batch. Defaults to 128.

    Returns:
        List[List[float]]: A list of embeddings, where each embedding is a list of floating point numbers.
    """
    embedding_list = []

    pbar = tqdm(total=len(txt_list))

    while txt_list:
        batch = txt_list[:batch_size]
        embeddings = model.encode(batch)
        embedding_list.extend(embeddings)
        del txt_list[:batch_size]

        pbar.update(len(batch))

        torch.cuda.empty_cache()
        gc.collect()  
    
    pbar.close()
    return embedding_list

def get_openai_embedding(text_list: List[str], model: str, client: Any) -> List[List[float]]:
    """
    Generates embeddings for a list of input texts using the specified OpenAI model.

    Args:
        text_list (List[str]): A list of input texts to generate embeddings for.
        model (str): The OpenAI model identifier used to generate the embeddings.
        client (Any): An instance of the OpenAI API client used to request the embeddings.

    Returns:
        List[List[float]]: A list of embeddings, where each embedding is a list of floating point numbers.
    """
    embedding_list = []

    for text in tqdm(text_list, total=len(text_list)):
        text = text.replace("\n", " ")
        embedding = client.embeddings.create(input = [text], model=model).data[0].embedding
        embedding_list.append(embedding)
    
    return embedding_list

# -------------------------------------------------------------------------#
# DATA PROCESSING FUNCTIONS
# -------------------------------------------------------------------------#

class MergeStrategy(Enum):
    FACT_EMB = 1
    POST_EMB = 2
    ALL = 3

def create_post_fact_df_with_ext_emb(df_facts: pd.DataFrame, df_posts: pd.DataFrame, df_mapping: pd.DataFrame, 
    fact_orig_emb_pth: str, fact_eng_emb_pth: str, post_l1_emb_pth: str, post_l2_emb_pth: str,
    strategy: MergeStrategy) -> pd.DataFrame:
    """
    Create a combined dataframe of posts, facts, and their respective embeddings based on the specified strategy.

    Args:
        df_facts (pd.DataFrame): DataFrame containing fact information.
        df_posts (pd.DataFrame): DataFrame containing post information.
        df_mapping (pd.DataFrame): Mapping DataFrame to relate posts and facts.
        fact_orig_emb_pth (str): Path to the pickled DataFrame containing original embeddings for facts.
        fact_eng_emb_pth (str): Path to the pickled DataFrame containing English embeddings for facts.
        post_l1_emb_pth (str): Path to the pickled DataFrame containing first-level embeddings for posts.
        post_l2_emb_pth (str): Path to the pickled DataFrame containing second-level embeddings for posts.
        strategy (MergeStrategy): Strategy to determine how to merge the data.

    Returns:
        pd.DataFrame: Combined DataFrame with post, fact, and respective embedding information.    
    """
    
    if strategy == MergeStrategy.FACT_EMB:
        # Load and merge the embeddings for FACTS
        fact_orig_emb_df = pd.read_pickle(fact_orig_emb_pth)
        fact_orig_emb_df = fact_orig_emb_df.rename(columns={'gpt_large_emb': 'orig_fact_large_emb', 'gpt_small_emb': 'orig_fact_small_emb'})
        df = df_facts.merge(fact_orig_emb_df, how='left')
        del fact_orig_emb_df
        gc.collect()
        
        fact_eng_emb_df = pd.read_pickle(fact_eng_emb_pth)
        fact_eng_emb_df = fact_eng_emb_df.rename(columns={'gpt_large_emb': 'eng_fact_large_emb', 'gpt_small_emb': 'eng_fact_small_emb'})
        df = df.merge(fact_eng_emb_df, how='left')
        del fact_eng_emb_df
        gc.collect()
    
    elif strategy == MergeStrategy.POST_EMB:
        # Load and merge the embeddings for POSTS
        post_l1_emb_df = pd.read_pickle(post_l1_emb_pth)
        post_l1_emb_df = post_l1_emb_df.rename(columns={'gpt_large_emb': 'l1_post_large_emb', 'gpt_small_emb': 'l1_post_small_emb'})
        df = df_posts.merge(post_l1_emb_df, how='left')
        del post_l1_emb_df
        gc.collect()
        
        post_l2_emb_df = pd.read_pickle(post_l2_emb_pth)
        post_l2_emb_df = post_l2_emb_df.rename(columns={'gpt_large_emb': 'l2_post_large_emb', 'gpt_small_emb': 'l2_post_small_emb'})
        df = df.merge(post_l2_emb_df, how='left')
        gc.collect()
    
    elif strategy == MergeStrategy.ALL:
        df = pd.merge(df_mapping, df_facts, how='left')
        df = df.merge(df_posts, how='left')
        
        # Load and merge the embeddings for FACTS
        fact_orig_emb_df = pd.read_pickle(fact_orig_emb_pth)
        fact_orig_emb_df = fact_orig_emb_df.rename(columns={'gpt_large_emb': 'orig_fact_large_emb', 'gpt_small_emb': 'orig_fact_small_emb'})
        df = df.merge(fact_orig_emb_df, how='left')
        del fact_orig_emb_df
        gc.collect()
        
        fact_eng_emb_df = pd.read_pickle(fact_eng_emb_pth)
        fact_eng_emb_df = fact_eng_emb_df.rename(columns={'gpt_large_emb': 'eng_fact_large_emb', 'gpt_small_emb': 'eng_fact_small_emb'})
        df = df.merge(fact_eng_emb_df, how='left')
        del fact_eng_emb_df
        gc.collect()
        
        # Load and merge the embeddings for POSTS
        post_l1_emb_df = pd.read_pickle(post_l1_emb_pth)
        post_l1_emb_df = post_l1_emb_df.rename(columns={'gpt_large_emb': 'l1_post_large_emb', 'gpt_small_emb': 'l1_post_small_emb'})
        df = df.merge(post_l1_emb_df, how='left')
        del post_l1_emb_df
        gc.collect()
        
        post_l2_emb_df = pd.read_pickle(post_l2_emb_pth)
        post_l2_emb_df = post_l2_emb_df.rename(columns={'gpt_large_emb': 'l2_post_large_emb', 'gpt_small_emb': 'l2_post_small_emb'})
        df = df.merge(post_l2_emb_df, how='left')
        gc.collect()
    
    return df


class TextType(Enum):
    POST_L1 = 1
    POST_L2 = 2
    FACT_ORIG = 3
    FACT_ENG = 4

def create_inference_dataset(df_facts: pd.DataFrame, df_posts: pd.DataFrame, df_mapping: pd.DataFrame, 
                             fact_orig_emb_pth: str, fact_eng_emb_pth: str, post_l1_emb_pth: str, post_l2_emb_pth: str,
                             txt_type: TextType) -> Tuple[List[str], List[Any], List[Any]]:
    """
    Generates a dataset for inference, which outputs text, small embedding, and large embedding
    for either posts or facts based on the specified embedding type.

    Args:
        df_facts (pd.DataFrame): DataFrame containing fact-check data.
        df_posts (pd.DataFrame): DataFrame containing posts data.
        df_mapping (pd.DataFrame): DataFrame containing mapping data between posts and facts.
        fact_orig_emb_pth (str): Path to the original fact embeddings.
        fact_eng_emb_pth (str): Path to the English fact embeddings.
        post_l1_emb_pth (str): Path to the post embeddings for language l1.
        post_l2_emb_pth (str): Path to the post embeddings for language l2.
        txt_type (TextType): The type of embeddings to output (Post L1, Post L2, Fact Original, or Fact English).

    Returns:
        Tuple[List[str], List[Any], List[Any]]:
            - A list of texts corresponding to the selected type (post or fact).
            - A list of small embeddings corresponding to the texts.
            - A list of large embeddings corresponding to the texts.
    """
    texts = []
    small_embeddings = []
    large_embeddings = []
        
    if txt_type == TextType.POST_L1:
        df = create_post_fact_df_with_ext_emb(df_facts, df_posts, df_mapping, fact_orig_emb_pth, fact_eng_emb_pth, post_l1_emb_pth, post_l2_emb_pth, strategy=MergeStrategy.POST_EMB)
        texts = df['post_l1'].tolist()
        small_embeddings = df['l1_post_small_emb'].tolist()
        large_embeddings = df['l1_post_large_emb'].tolist()

    elif txt_type == TextType.POST_L2:
        df = create_post_fact_df_with_ext_emb(df_facts, df_posts, df_mapping, fact_orig_emb_pth, fact_eng_emb_pth, post_l1_emb_pth, post_l2_emb_pth, strategy=MergeStrategy.POST_EMB)
        texts = df['post_l2'].tolist()
        small_embeddings = df['l2_post_small_emb'].tolist()
        large_embeddings = df['l2_post_large_emb'].tolist()

    elif txt_type == TextType.FACT_ORIG:
        df = create_post_fact_df_with_ext_emb(df_facts, df_posts, df_mapping, fact_orig_emb_pth, fact_eng_emb_pth, post_l1_emb_pth, post_l2_emb_pth, strategy=MergeStrategy.FACT_EMB)
        texts = df['facts_orig'].tolist()
        small_embeddings = df['orig_fact_small_emb'].tolist()
        large_embeddings = df['orig_fact_large_emb'].tolist()

    elif txt_type == TextType.FACT_ENG:
        df = create_post_fact_df_with_ext_emb(df_facts, df_posts, df_mapping, fact_orig_emb_pth, fact_eng_emb_pth, post_l1_emb_pth, post_l2_emb_pth, strategy=MergeStrategy.FACT_EMB)
        texts = df['facts_eng'].tolist()
        small_embeddings = df['eng_fact_small_emb'].tolist()
        large_embeddings = df['eng_fact_large_emb'].tolist()

    return texts, small_embeddings, large_embeddings


def create_train_dataset(df_facts: pd.DataFrame, df_posts: pd.DataFrame, df_mapping: pd.DataFrame, 
                         fact_orig_emb_pth: str, fact_eng_emb_pth: str,
                         post_l1_emb_pth: str, post_l2_emb_pth: str,
                         train_size: float = 1) -> Tuple[List[Tuple[str, str]], List[Tuple[Any, Any]], List[Tuple[Any, Any]]]:
    """
    Generates a dataset for multiple negative ranking loss training by pairing posts and facts
    with associated embeddings. The dataset includes pairs of anchor posts and positive facts,
    as well as their corresponding embeddings (both small and large versions).

    Args:
        df_facts (pd.DataFrame): DataFrame containing fact-check data.
        df_posts (pd.DataFrame): DataFrame containing posts data.
        df_mapping (pd.DataFrame): DataFrame containing mapping data between posts and facts.
        fact_orig_emb_pth (str): Path to the original fact embeddings.
        fact_eng_emb_pth (str): Path to the English fact embeddings.
        post_l1_emb_pth (str): Path to the post embeddings for language l1.
        post_l2_emb_pth (str): Path to the post embeddings for language l2.

    Returns:
        Tuple[List[Tuple[str, str]], List[Tuple[Any, Any]], List[Tuple[Any, Any]]]:
            - A list of text pairs (anchor post, positive fact) for training.
            - A list of small embedding pairs corresponding to the text pairs.
            - A list of large embedding pairs corresponding to the text pairs.
    """
    df = create_post_fact_df_with_ext_emb(df_facts, df_posts, df_mapping, fact_orig_emb_pth, fact_eng_emb_pth, post_l1_emb_pth, post_l2_emb_pth, strategy=MergeStrategy.ALL)
    if train_size <1:
        df = df.sample(frac = train_size, random_state=4727).reset_index(drop = True)

    anchor_post = []
    positive_fact = []

    anchor_post_emb_large = []
    pos_fact_emb_large = []

    anchor_post_emb_small = []
    pos_fact_emb_small = []

    for ix, row in tqdm(df.iterrows()):

        if len(set(row['post_text'][:2])) == 1:
            anchor_post.append(row['post_text'][:1])
            anchor_post_emb_large.append([row['l1_post_large_emb']])    
            anchor_post_emb_small.append([row['l1_post_small_emb']])  

        if len(set(row['post_text'][:2])) == 2:
            anchor_post.append(row['post_text'][:2])
            anchor_post_emb_large.append([row['l1_post_large_emb'], row['l2_post_large_emb']])    
            anchor_post_emb_small.append([row['l1_post_small_emb'], row['l2_post_small_emb']])   

        if len(set(row['fact_claim'][:2])) == 1:
            positive_fact.append(row['fact_claim'][:1])
            pos_fact_emb_large.append([row['orig_fact_large_emb']])
            pos_fact_emb_small.append([row['orig_fact_small_emb']])

        if len(set(row['fact_claim'][:2])) == 2:
            positive_fact.append(row['fact_claim'][:2])
            pos_fact_emb_large.append([row['orig_fact_large_emb'], row['eng_fact_large_emb']])
            pos_fact_emb_small.append([row['orig_fact_small_emb'], row['eng_fact_small_emb']])

    dataset = [(a, p) for a_list, p_list in zip(anchor_post, positive_fact) for a in a_list for p in p_list]
    emb_small_list = [(a, p) for a_list, p_list in zip(anchor_post_emb_small, pos_fact_emb_small) for a in a_list for p in p_list]
    emb_large_list = [(a, p) for a_list, p_list in zip(anchor_post_emb_large, pos_fact_emb_large) for a in a_list for p in p_list]

    return dataset, emb_small_list, emb_large_list

# -------------------------------------------------------------------------#
# DATA PROCESSING FUNCTIONS
# -------------------------------------------------------------------------#
def process_sample(batch):
    # Separate the different components of the batch
    input_ids = [item['input_ids'] for item in batch if ((item['ext_embedding1'].shape != torch.Size([])) and (item['ext_embedding2'].shape != torch.Size([])))]
    attention_masks = [item['attention_mask'] for item in batch if ((item['ext_embedding1'].shape != torch.Size([])) and (item['ext_embedding2'].shape != torch.Size([])))]
    ext_embeddings1 = [item['ext_embedding1'] for item in batch if ((item['ext_embedding1'].shape != torch.Size([])) and (item['ext_embedding2'].shape != torch.Size([])))]
    ext_embeddings2 = [item['ext_embedding2'] for item in batch if ((item['ext_embedding1'].shape != torch.Size([])) and (item['ext_embedding2'].shape != torch.Size([])))]

    # Pad the sequences
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_masks = pad_sequence(attention_masks, batch_first=True, padding_value=0)

    # Stack the embeddings
    ext_embeddings1 = torch.stack(ext_embeddings1)
    ext_embeddings2 = torch.stack(ext_embeddings2)

    return {
        'input_ids': input_ids,
        'attention_mask': attention_masks,
        'ext_embedding1': ext_embeddings1,
        'ext_embedding2': ext_embeddings2
    }

def collate_inf_fn(batch):
    result = process_sample(batch)
    return result

def collate_fn_train(batch):
    # Separate the anchor and positive samples
    anchor_samples = [item[0] for item in batch]
    positive_samples = [item[1] for item in batch]

    # Process anchor and positive samples
    anchor_batch = process_sample(anchor_samples)
    positive_batch = process_sample(positive_samples)

    return [anchor_batch, positive_batch]

class PairDatasetWithEmbeddings_for_train(Dataset):
    """
    A PyTorch Dataset for loading paired anchor (post) and positive (fact) examples with associated external embeddings for use in training.

    Args:
        data (List[Tuple[str, str]]): A list of text pairs where each pair consists of an anchor (post) and 
            a positive (fact).
        tokenizer (Any): A tokenizer instance (e.g., from HuggingFace) used to tokenize the input texts.
        external_embeddings1 (List[Tuple[Any, Any]]): A list of external embeddings (e.g., GPT-small embeddings) 
            for the anchor and positive pairs.
        external_embeddings2 (List[Tuple[Any, Any]]): A list of external embeddings (e.g., GPT-large embeddings) 
            for the anchor and positive pairs.
        max_length (int, optional): The maximum length for tokenized input sequences. Defaults to 512.

    Returns:
        Dict: A dictionary containing tokenized inputs and external embeddings for the anchor and positive pairs.
            - 'input_ids': Tensor of input token IDs.
            - 'attention_mask': Tensor of attention masks.
            - 'ext_embedding1': The first external embedding (e.g., GPT-small embedding).
            - 'ext_embedding2': The second external embedding (e.g., GPT-large embedding).
    """
    def __init__(self, data, tokenizer, external_embeddings1, external_embeddings2, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.external_embeddings1 = external_embeddings1
        self.external_embeddings2 = external_embeddings2
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        anchor, positive = self.data[idx]
        anchor_encoded = self.tokenizer.encode_plus(anchor, truncation=True, max_length=self.max_length, return_tensors='pt')
        positive_encoded = self.tokenizer.encode_plus(positive, truncation=True, max_length=self.max_length, return_tensors='pt')

        # anchor_encoded = self.tokenizer.encode_plus(anchor, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        # positive_encoded = self.tokenizer.encode_plus(positive, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')

        anchor_embedding1 = torch.tensor(self.external_embeddings1[idx][0])
        positive_embedding1 = torch.tensor(self.external_embeddings1[idx][1])
        anchor_embedding2 = torch.tensor(self.external_embeddings2[idx][0])
        positive_embedding2 = torch.tensor(self.external_embeddings2[idx][1])
        return [
                {
                    'input_ids': anchor_encoded['input_ids'].reshape(-1), 
                    'attention_mask': anchor_encoded['attention_mask'].reshape(-1), 
                    'ext_embedding1': anchor_embedding1, 
                    'ext_embedding2': anchor_embedding2
                },
                {
                    'input_ids': positive_encoded['input_ids'].reshape(-1), 
                    'attention_mask': positive_encoded['attention_mask'].reshape(-1), 
                    'ext_embedding1': positive_embedding1, 
                    'ext_embedding2': positive_embedding2
                }
               ]
    
class PairDatasetWithEmbeddings_for_inf(Dataset):
    """
    A PyTorch Dataset for inference that returns tokenized text and associated external embeddings.

    Args:
        txt_list (List[str]): A list of input texts for which the embeddings are generated.
        tokenizer (Any): A tokenizer instance (e.g., from HuggingFace) used to tokenize the input texts.
        external_embeddings1 (List[Any]): A list of external embeddings (e.g., GPT-small embeddings) 
            corresponding to the input texts.
        external_embeddings2 (List[Any]): A list of external embeddings (e.g., GPT-large embeddings) 
            corresponding to the input texts.
        max_length (int, optional): The maximum length for tokenized input sequences. Defaults to 512.

    Returns:
        Dict[str, Any]: A dictionary containing tokenized input text and corresponding external embeddings:
            - 'input_ids': Tensor of input token IDs.
            - 'attention_mask': Tensor of attention masks.
            - 'ext_embedding1': The first external embedding (e.g., GPT-small embedding).
            - 'ext_embedding2': The second external embedding (e.g., GPT-large embedding).
    """
    def __init__(self, txt_list, tokenizer, external_embeddings1, external_embeddings2, max_length=512):
        self.data = txt_list
        self.tokenizer = tokenizer
        self.external_embeddings1 = external_embeddings1
        self.external_embeddings2 = external_embeddings2
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        txt = self.data[idx]
        # txt_encoded = self.tokenizer.encode_plus(txt, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        txt_encoded = self.tokenizer.encode_plus(txt, truncation=True, max_length=self.max_length, return_tensors='pt')
        txt_embedding1 = torch.tensor(self.external_embeddings1[idx])
        txt_embedding2 = torch.tensor(self.external_embeddings2[idx])

        return {'input_ids': txt_encoded['input_ids'].reshape(-1), 
                'attention_mask': txt_encoded['attention_mask'].reshape(-1), 
                'ext_embedding1': txt_embedding1, 
                'ext_embedding2': txt_embedding2}
    