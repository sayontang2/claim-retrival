import logging
import torch
from torch.utils.data import DataLoader
from transformers import (AdamW, AutoTokenizer, 
                          get_linear_schedule_with_warmup,)
from transformers import logging as trg_logging
from torchmetrics.functional.pairwise import pairwise_cosine_similarity
from tqdm import tqdm
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Custom imports
from model import (ModifiedSentenceTransformer, FeatureSet, AggregationMethod)
from dataset import (PairDatasetWithEmbeddings_for_train, get_raw_data_df, collate_fn_train,
                    process_facts_df, process_posts_df, create_train_dataset)
from evaluate import evaluate
from utils import save_model, load_model, set_seed
trg_logging.set_verbosity_warning()
import pdb


def setup_config():
    """Handle all configuration and hyperparameter setup."""

    config = {
        # Model Hyperparameters
        'feature_type' : FeatureSet.SBERT_EXT1,
        'agg_type' : AggregationMethod.LINEAR,
        'nheads' : 8, 
        'model_name' : 'sentence-transformers/use-cmlm-multilingual',
        'external_dim1' : 1536, 
        'external_dim2' : 3072,
        'device' : 'cuda:0',
        'seed' : 447,
        
        # Data Paths
        'train_fact_path' : './in_data/fact_checks.csv',
        'train_post_path' : './in_data/posts.csv',
        'train_post2fact_mapping' : './in_data/fact_check_post_mapping.csv',

        'train_fact_orig_emb_path' : './openai-op/orig-fact.pkl',
        'train_fact_eng_emb_path' : './openai-op/eng-fact.pkl',
        'train_post_l1_emb_path' : './openai-op/l1-post.pkl',
        'train_post_l2_emb_path' : './openai-op/l2-post.pkl',

        # 'train_fact_path' : './sample_data/trial_fact_checks.csv',
        # 'train_post_path' : './sample_data/trial_posts.csv',
        # 'train_post2fact_mapping' : './sample_data/trial_data_mapping.csv',

        # 'train_fact_orig_emb_path' : './openai-op/eval_orig-fact.pkl',
        # 'train_fact_eng_emb_path' : './openai-op/eval_eng-fact.pkl',
        # 'train_post_l1_emb_path' : './openai-op/eval_l1-post.pkl',
        # 'train_post_l2_emb_path' : './openai-op/eval_l2-post.pkl',

        'eval_fact_path' : './sample_data/trial_fact_checks.csv',
        'eval_post_path' : './sample_data/trial_posts.csv',
        'eval_post2fact_mapping' : './sample_data/trial_data_mapping.csv',

        'eval_fact_orig_emb_path' : './openai-op/eval_orig-fact.pkl',
        'eval_fact_eng_emb_path' : './openai-op/eval_eng-fact.pkl',
        'eval_post_l1_emb_path' : './openai-op/eval_l1-post.pkl',
        'eval_post_l2_emb_path' : './openai-op/eval_l2-post.pkl',

        # Training Hyperparameters
        'loss_scale' : 20,
        'train_batch_size' : 16,
        'accumulation_steps': 1,
        'eval_batch_size' :4,
        'max_seq_len' : 312,
        'n_epochs' : 5,
        'wt_decay' : 1e-8,
        'lr' : 1e-4,
        'K': 5,

        # Logging & Saving
        'log_path': 'exp_log.log',
        'checkpoint_dir': './checkpoints/',
        'model_checkpointing': False,
        'load_checkpoint': False,
        'checkpoint_path': './checkpoints/checkpoint_epoch_1.pth',
        'loss_log_step': 5
    }
    return config

def prepare_training_data(config, tokenizer):
    """Prepare the training and training datasets."""

    # Load data
    facts, posts, post2fact_mapping = get_raw_data_df(config['train_fact_path'], config['train_post_path'], config['train_post2fact_mapping'])
    facts = process_facts_df(facts)
    posts = process_posts_df(posts)
    
    # Create dataset
    an_po_pair_dataset, an_po_small_ext_emb, an_po_large_ext_emb = create_train_dataset(
                                                                        facts, posts, post2fact_mapping,
                                                                        config['train_fact_orig_emb_path'], config['train_fact_eng_emb_path'],
                                                                        config['train_post_l1_emb_path'], config['train_post_l2_emb_path']
                                                                    )
    dataset = PairDatasetWithEmbeddings_for_train(an_po_pair_dataset, tokenizer, an_po_small_ext_emb, an_po_large_ext_emb)
    return dataset

def prepare_eval_data(config):
    """Prepare evaluation dataset"""

    # Load data
    facts, posts, post2fact_mapping = get_raw_data_df(config['eval_fact_path'], config['eval_post_path'], config['eval_post2fact_mapping'])

    facts = process_facts_df(facts)
    posts = process_posts_df(posts)
    
    return posts, facts, post2fact_mapping

def setup_logging(log_path):
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()]
    )
    logger = logging.getLogger(__name__)
    return logger

def initialize_model(config):
    """Initialize model tokenizer."""

    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
    
    # num_heads arg is used only when aggregation_method = LINEAR
    model = ModifiedSentenceTransformer(
        model_name=config['model_name'], external_dim1=config['external_dim1'], external_dim2=config['external_dim2'],
        feature_set=config['feature_type'], aggregation_method=config['agg_type'], num_heads=config['nheads']
    ).to(config['device'])

    return model, tokenizer

def initalize_optimizer(config, model, steps_per_batch):
    """Initialize optimizer and scheduler.""" 
    no_decay = ['bias' , 'gamma', 'beta']
    optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': config['wt_decay']},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    optimizer = AdamW(optimizer_grouped_parameters, lr=config['lr'])

    # setup warmup for first ~10% of steps
    total_steps = (config['n_epochs'] * steps_per_batch) // config['accumulation_steps']
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0.1 * total_steps, num_training_steps=total_steps)
    
    return optimizer, scheduler

def forward_pass(model, batch, config):
    """Forward pass for model."""

    batch[0] = {k:v.to(config['device']) for k, v in batch[0].items()}
    batch[1] = {k:v.to(config['device']) for k, v in batch[1].items()}
    
    anchor_repr = model(batch[0])
    positive_repr = model(batch[1])
    
    return anchor_repr, positive_repr

def compute_loss(anchor_repr, positive_repr, config):
    """Compute loss for anchor-positive pairs."""

    loss_func = torch.nn.CrossEntropyLoss()
    scores = pairwise_cosine_similarity(anchor_repr, positive_repr)
    labels = torch.arange(len(scores), device=scores.device)
    loss = loss_func(scores * config['loss_scale'], labels)

    return loss

# def train_one_epoch(epoch, model, optimizer, scheduler, train_loader, config):
#     """Training logic for one epoch with gradient accumulation."""
#     model.train()
#     loop = tqdm(train_loader, leave=True)
        
#     optimizer.zero_grad()  # Initialize the gradients

#     for ix, batch in enumerate(loop):
#         loop.set_description(f'Epoch {epoch}')
#         # loop.set_postfix(loss=loss.item() * accumulation_steps)  # Unscaled loss for logging


def train_one_epoch(epoch, model, optimizer, scheduler, train_loader, config):
    """Training logic for one epoch with gradient accumulation."""
    model.train()
    loop = tqdm(train_loader, leave=True)
    
    # Number of steps after which gradients are accumulated
    accumulation_steps = config['accumulation_steps']
    
    optimizer.zero_grad()  # Initialize the gradients

    for ix, batch in enumerate(loop):
        # Forward pass
        anchor_repr, positive_repr = forward_pass(model, batch, config)
        
        # Compute loss
        loss = compute_loss(anchor_repr, positive_repr, config)
    
        # Normalize loss by accumulation steps
        loss = loss / accumulation_steps
        
        # Backpropagate loss
        loss.backward()
        
        # Update the parameters and scheduler every `accumulation_steps` steps
        if (ix + 1) % accumulation_steps == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()  # Reset gradients after update

            # Log the batch loss if required
            if (config['loss_log_step'] > 0) and (((ix + 1) / accumulation_steps) % config['loss_log_step'] == 0):
                logging.info(f'Epoch {epoch} - Step {(ix + 1) / accumulation_steps}: Loss = {loss.item() * accumulation_steps}')
    
        loop.set_description(f'Epoch {epoch}')
        loop.set_postfix(loss=loss.item() * accumulation_steps)  # Unscaled loss for logging
    
    logging.info(f'Epoch {epoch}: Last Batch Loss = {loss.item() * accumulation_steps}')  # Log the last batch loss


def evaluate_model(config, epoch, model, tokenizer, posts, facts, post2fact_mapping):
    """Perform evaluation after each epoch."""
    model.eval()
    result = evaluate(model, tokenizer, config['device'], posts, facts, post2fact_mapping, 
                            config['eval_fact_orig_emb_path'], config['eval_fact_eng_emb_path'], 
                            config['eval_post_l1_emb_path'], config['eval_post_l2_emb_path'],
                            config['eval_batch_size'], config['K'])
    
    logging.info(f'Evaluation result at epoch {epoch}:\n\t'+ result.to_string().replace('\n', '\n\t'))

def main():
    # ----------------------------------------------------
    # Configuration and argument parsing
    # ----------------------------------------------------
    config = setup_config()
    config['log_path'] = f'./logs/exp_seed-{config["seed"]}_feat-{config["feature_type"]}.log'
    set_seed(config['seed'])
    
    # Setup logging
    setup_logging(config['log_path'])
    
    # ----------------------------------------------------
    # Model and tokenizer initialization
    # ----------------------------------------------------
    model, tokenizer = initialize_model(config)

    # ----------------------------------------------------
    # Load datasets and create dataloaders
    # ----------------------------------------------------
    train_dataset = prepare_training_data(config, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size = config['train_batch_size'], shuffle = True, 
                              collate_fn = collate_fn_train, num_workers = 4)
    eval_post, eval_fact, eval_post2facct_mapping = prepare_eval_data(config)

    # ----------------------------------------------------
    # Optimizer and scheduler initialization
    # ----------------------------------------------------
    steps_per_batch = int(len(train_dataset) / config['train_batch_size'])
    optimizer, scheduler = initalize_optimizer(config, model, steps_per_batch)

    # ----------------------------------------------------
    # Optionally, load a pre-trained model
    # ----------------------------------------------------
    if config['load_checkpoint']:
        start_epoch = load_model(model, optimizer, scheduler, config['checkpoint_path'])
    else:
        start_epoch = 0

    # ----------------------------------------------------
    # Training Loop
    # ----------------------------------------------------
    for epoch in range(start_epoch, config['n_epochs']):
        # Evaluation after each epoch
        evaluate_model(config, epoch, model, tokenizer, eval_post, eval_fact, eval_post2facct_mapping)

        # Train model for one epoch
        train_one_epoch(epoch, model, optimizer, scheduler, train_loader, config)

        # Save checkpoint
        if config['model_checkpointing'] == True:
            save_model(model, optimizer, scheduler, epoch, config['checkpoint_dir'])
            logging.info(f'Model saved to {config["checkpoint_dir"]}')

    # Final evaluation at the end
    evaluate_model(config, (epoch + 1), model, tokenizer, eval_post, eval_fact, eval_post2facct_mapping)
    
if __name__ == '__main__':
    main()