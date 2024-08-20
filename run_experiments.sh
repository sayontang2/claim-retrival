#!/bin/bash

# Define the FeatureSet and AggregationMethod enum values (using integer values)
feature_types=(1 2 3 4)  # 1 for SBERT_ONLY, 2 for SBERT_EXT1, etc.
agg_types=(1 2)  # 1 for LINEAR, 2 for ATTENTION

# Fixed values for batch_size and learning rate
batch_size=16
lr=1e-4

# nheads values only relevant for ATTENTION (agg_type=2)
nheads_list=(4 8)

# Loop over all combinations of parameters
for feature_type in "${feature_types[@]}"; do
  for agg_type in "${agg_types[@]}"; do

    # Check if agg_type is ATTENTION (2), if so, iterate over nheads
    if [ "$agg_type" -eq 2 ]; then
      for nheads in "${nheads_list[@]}"; do
        # Launch the experiment with nheads iteration
        echo "Running experiment with feature_type=$feature_type, agg_type=$agg_type (ATTENTION), nheads=$nheads, batch_size=$batch_size, lr=$lr"
        python src/main.py --feature_type $feature_type --agg_type $agg_type --nheads $nheads --train_batch_size $batch_size --lr $lr
      done
    else
      # If agg_type is LINEAR (1), run the experiment without iterating over nheads
      echo "Running experiment with feature_type=$feature_type, agg_type=$agg_type (LINEAR), batch_size=$batch_size, lr=$lr"
      python src/main.py --feature_type $feature_type --agg_type $agg_type --nheads 0 --train_batch_size $batch_size --lr $lr
    fi

  done
done