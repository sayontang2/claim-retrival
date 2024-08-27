#!/bin/bash

# Define the FeatureSet, AggregationMethod, and training_data_size enum values (using integer values)
feature_types=(1 2 3 4)  # 1 for SBERT_ONLY, 2 for SBERT_EXT1, etc.
agg_types=(1 2)  # 1 for LINEAR, 2 for ATTENTION
train_sizes=(0.01 0.05 0.1)  # Different fractions of the training data to use

# nheads values only relevant for ATTENTION (agg_type=2)
nheads_list=(4 8)

# Loop over all combinations of parameters
for feature_type in "${feature_types[@]}"; do
  for agg_type in "${agg_types[@]}"; do
    for train_size in "${train_sizes[@]}"; do

      # Check if agg_type is ATTENTION (2), if so, iterate over nheads
      if [ "$agg_type" -eq 2 ]; then
        for nheads in "${nheads_list[@]}"; do
          # Launch the experiment with nheads and training data size iteration
          echo "Running experiment with feature_type=$feature_type, agg_type=$agg_type (ATTENTION), nheads=$nheads, train_size=$train_size"
          python src/main.py --feature_type $feature_type --agg_type $agg_type --nheads $nheads --train_size $train_size
        done
      else
        # If agg_type is LINEAR (1), run the experiment without iterating over nheads
        echo "Running experiment with feature_type=$feature_type, agg_type=$agg_type (LINEAR), train_size=$train_size"
        python src/main.py --feature_type $feature_type --agg_type $agg_type --nheads 0 --train_size $train_size
      fi

    done
  done
done
