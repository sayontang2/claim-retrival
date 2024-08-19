import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from enum import Enum

class FeatureSet(Enum):
    SBERT_ONLY = 1
    SBERT_EXT1 = 2
    SBERT_EXT2 = 3
    SBERT_EXT1_EXT2 = 4

class AggregationMethod(Enum):
    LINEAR = 1
    ATTENTION = 2

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
    
    def forward(self, query, key, value):
        attn_output, _ = self.mha(query, key, value)
        return attn_output

class ModifiedSentenceTransformer(SentenceTransformer):
    """
    A modified version of the SentenceTransformer model that incorporates external embeddings
    and combines them with the SBERT embeddings using either linear weight aggregation or
    multi-headed self-attention.

    Args:
        model_name (str): The name of the SentenceTransformer model to be loaded.
        external_dim1 (int): The dimensionality of the first external embedding.
        external_dim2 (int): The dimensionality of the second external embedding.
        feature_set (FeatureSet): An enum specifying which combination of features to use.
            Default is FeatureSet.SBERT_EXT1_EXT2 (using all features).
        aggregation_method (AggregationMethod): Method to aggregate features. 
            Can be LINEAR or ATTENTION. Default is LINEAR.
        num_heads (int): Number of attention heads for multi-headed self-attention. 
            Only used if aggregation_method is ATTENTION. Default is 8.

    Attributes:
        sbert (SentenceTransformer): The base SBERT model.
        target_dim (int): The target dimensionality for all embeddings.
        linear_ext1 (nn.Linear): Linear layer for transforming the first external embedding.
        linear_ext2 (nn.Linear): Linear layer for transforming the second external embedding.
        feature_set (FeatureSet): The selected feature set for this instance.
        aggregation_method (AggregationMethod): The selected aggregation method.
        attention (MultiHeadAttention): Multi-headed self-attention module for feature aggregation.
        final_linear (nn.Linear): Final linear layer for aggregation or projection.

    Methods:
        forward(batch): Forward pass to compute the combined sentence embedding.

    Forward Pass Args:
        batch (Dict[str, Any]): A batch of inputs containing the following keys:
            - 'input_ids': Tensor of input token IDs for SBERT.
            - 'attention_mask': Tensor of attention masks for SBERT.
            - 'ext_embedding1': Tensor of the first external embedding (required if using EXT1).
            - 'ext_embedding2': Tensor of the second external embedding (required if using EXT2).

    Returns:
        torch.Tensor: The final sentence embedding after combining the selected features.

    Usage:
        # Using all features with linear aggregation (default behavior)
        model_linear = ModifiedSentenceTransformer("bert-base-nli-mean-tokens", 768, 1024)

        # Using all features with attention-based aggregation
        model_attention = ModifiedSentenceTransformer("bert-base-nli-mean-tokens", 768, 1024, aggregation_method=AggregationMethod.ATTENTION)

        # Using SBERT and EXT1 with linear aggregation
        model_sbert_ext1_linear = ModifiedSentenceTransformer("bert-base-nli-mean-tokens", 768, 1024, feature_set=FeatureSet.SBERT_EXT1)

        # Using SBERT and EXT1 with attention-based aggregation
        model_sbert_ext1_attention = ModifiedSentenceTransformer("bert-base-nli-mean-tokens", 768, 1024, feature_set=FeatureSet.SBERT_EXT1, 
                                                                  aggregation_method=AggregationMethod.ATTENTION)

    Note:
        The model adapts its architecture based on the selected feature_set and aggregation_method, 
        allowing for flexible feature selection and aggregation strategies.
    """

    def __init__(self, model_name, external_dim1, external_dim2, 
                 feature_set=FeatureSet.SBERT_EXT1_EXT2, aggregation_method=AggregationMethod.LINEAR, num_heads=8):
        super(ModifiedSentenceTransformer, self).__init__(model_name)
        self.sbert = SentenceTransformer(model_name)
        self.target_dim = self.sbert.get_sentence_embedding_dimension()
        self.linear_ext1 = nn.Linear(external_dim1, self.target_dim)
        self.linear_ext2 = nn.Linear(external_dim2, self.target_dim)
        self.feature_set = feature_set
        self.aggregation_method = aggregation_method
        
        if self.aggregation_method == AggregationMethod.ATTENTION:
            self.attention = MultiHeadAttention(self.target_dim, num_heads)
            self.final_linear = nn.Linear(self.target_dim, self.target_dim)

        else:  # LINEAR
            if self.feature_set == FeatureSet.SBERT_ONLY:
                self.final_linear = nn.Identity()
            elif self.feature_set in [FeatureSet.SBERT_EXT1, FeatureSet.SBERT_EXT2]:
                self.final_linear = nn.Linear(self.target_dim * 2, self.target_dim)
            else:  # FeatureSet.SBERT_EXT1_EXT2
                self.final_linear = nn.Linear(self.target_dim * 3, self.target_dim)

    def forward(self, batch):
        sbert_embedding = self.sbert({'input_ids': batch['input_ids'], 'attention_mask': batch['attention_mask']})['sentence_embedding']
        
        if self.feature_set == FeatureSet.SBERT_ONLY:
            return sbert_embedding
        
        embeddings = [sbert_embedding]
        
        if self.feature_set in [FeatureSet.SBERT_EXT1, FeatureSet.SBERT_EXT1_EXT2]:
            ext_embedding1_transformed = self.linear_ext1(batch['ext_embedding1'])
            embeddings.append(ext_embedding1_transformed)
        
        if self.feature_set in [FeatureSet.SBERT_EXT2, FeatureSet.SBERT_EXT1_EXT2]:
            ext_embedding2_transformed = self.linear_ext2(batch['ext_embedding2'])
            embeddings.append(ext_embedding2_transformed)
        
        if self.aggregation_method == AggregationMethod.ATTENTION:
            # Stack embeddings for key and value
            key_value = torch.stack(embeddings, dim=1)
            # Use SBERT embedding as query
            query = sbert_embedding.unsqueeze(1)
            # Apply multi-headed self-attention
            attn_output = self.attention(query, key_value, key_value)
            # Final linear projection
            final_embedding = self.final_linear(attn_output.squeeze(1))
        else:  # LINEAR
            concatenated_embedding = torch.cat(embeddings, dim=-1)
            final_embedding = self.final_linear(concatenated_embedding)
        
        return final_embedding
    

def calculate_mean_of_weights(model):
    """Calculate the mean of all the parameter weights in a model."""
    total_sum = 0.0
    total_params = 0
    
    # Iterate over all parameters
    for param in model.parameters():
        if param.requires_grad:  # Only consider trainable parameters
            total_sum += torch.sum(param.data)
            total_params += param.numel()  # Total number of elements (weights) in the parameter tensor

    # Calculate mean
    mean_weight = total_sum / total_params
    return mean_weight.item()

def calculate_mean_of_gradients(model):
    """Calculate the mean of gradients of all parameters in a model."""
    total_sum = 0.0
    total_params = 0
    
    # Iterate over all parameters
    for param in model.parameters():
        if param.grad is not None:  # Make sure that the gradient exists
            total_sum += torch.sum(param.grad)
            total_params += param.grad.numel()  # Total number of elements in the gradient tensor

    # Calculate mean
    mean_grad = total_sum / total_params if total_params > 0 else 0.0
    return mean_grad.item()