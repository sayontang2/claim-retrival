import os
import torch
import numpy as np
import random

def get_top_k_acc(pred_fact_id_matrix, true_fact_id_list, top_K):
    """
    Computes the top-K accuracy for a given set of predictions.

    This function evaluates the model's ability to correctly identify the true fact-check IDs 
    within the top-K most similar predictions, using cosine similarity or another similarity metric.
    For each true fact ID in `true_fact_id_list`, it checks if it is present within the top-1, top-2, ..., 
    up to top-K most similar fact IDs from `pred_fact_id_matrix`. The function then returns the average 
    accuracy for each of the top-K values.

    Args:
        pred_fact_id_matrix (numpy.ndarray): A 2D array of shape (n_samples, top_K), where each row contains 
                                             the predicted fact-check IDs ordered by similarity for a given post.
        true_fact_id_list (list): A list of true fact-check IDs corresponding to each sample (post).
        top_K (int): The number of top predictions to consider for the accuracy calculation (e.g., top-1, top-2, ..., top-K).

    Returns:
        numpy.ndarray: A 1D array of length `top_K`, where each element at index `k` contains the average 
                       top-k accuracy (i.e., the proportion of cases where the true fact-check ID was within 
                       the top-k predicted fact IDs).
    """

    result = []
    for true_id, pred_ids in zip(true_fact_id_list, pred_fact_id_matrix):
        correct_pred = [(true_id in pred_ids[:k]) for k in range(1, (top_K + 1))]
        result.append(correct_pred)
    result = np.array(result).mean(axis = 0)
    return result

def save_model(model, optimizer, scheduler, epoch, path):
    """
    Saves the model's state, optimizer's state, scheduler's state, and current epoch.

    Args:
    model (torch.nn.Module): The model to be saved.
    optimizer (torch.optim.Optimizer): The optimizer to be saved.
    scheduler (torch.optim.lr_scheduler): The scheduler to be saved.
    epoch (int): The current epoch.
    path (str): The directory to save the checkpoint.

    """
    if not os.path.exists(path):
        os.makedirs(path)
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': epoch
    }
    save_path = os.path.join(path, f'checkpoint_epoch_{epoch}.pth')
    torch.save(checkpoint, save_path)


def load_model(model, optimizer, scheduler, path):
    """
    Loads the model's state, optimizer's state, scheduler's state, and current epoch.

    Args:
    model (torch.nn.Module): The model to load the state into.
    optimizer (torch.optim.Optimizer): The optimizer to load the state into.
    scheduler (torch.optim.lr_scheduler): The scheduler to load the state into.
    path (str): The file path to the checkpoint.

    Returns:
    int: The epoch to resume from.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No checkpoint found at {path}")
    
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    epoch = checkpoint['epoch']
        
    return epoch

def set_seed(seed: int):
    """
    Set the seed for reproducibility.

    Args:
        seed (int): The seed value to be set.
    """
    # Set seed for random number generators
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If you are using multi-GPU.
    
    # Set deterministic behavior (if needed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"Seed set to {seed}")