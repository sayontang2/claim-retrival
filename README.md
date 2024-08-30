## About the Task

The SemEval shared task on Multilingual and Crosslingual Fact-Checked Claim Retrieval addresses the critical challenge of efficiently identifying previously fact-checked claims across multiple languages — a task that can be time-consuming for professional fact-checkers even within a single language and becomes much more difficult to perform manually when the claim and the fact-check may be in different languages. Given the global spread of disinformation narratives, the range of languages that one would need to cover not to miss existing fact-checks is vast.

Participants in the task will **develop systems to retrieve relevant fact-checked claims** for given social media posts, using the [MultiClaim dataset](https://zenodo.org/records/7737983) of over 200,000 fact-checked claims and 28,000 social media posts in 27 languages – supporting fact-checkers and researchers in their efforts to curb the spread of misinformation globally. Submissions will be evaluated in terms of two metrics, mean reciprocal rank and success-at-K and in two separate tracks: monolingual and crosslingual. The task not only pushes the boundaries of NLP and information retrieval, but it also has significant potential for real-world impact in the fight against misinformation.s

**Important links:**

* [Shared task webpage](https://kinit-sk.github.io/semeval_2025/);
* [DisAI project webpage](https://disai.eu/);
* [Original dataset paper](https://arxiv.org/abs/2305.07991) for additional details about the dataset;

## Sample Data

[[LINK TO THE SAMPLE DATA AT GITHUB]](https://github.com/kinit-sk/semeval_2025/tree/main/sample_data)

In this task, you are given social media posts (SMP), and a bunch of fact-checks (FC). The goal is to find the most relevant fact-checks for each social media post.

In this trial data, we have in total 50 SMP-FC pairs. Out of them, we have 10 eng-eng pairs and 40 pairs in 2 different languages (each has 20 examples, 10 for monolingual (e.g., kor-kor) and 10 for multilingual (e.g., kor-eng)).

Potential retrieval setup can be various. For example, for each SMP, the FC search pool can be limited to the target language, different languages, and different mix of languages.

In this task, we aim to evaluate in both monolingual and crosslingual setup. Common metrics include mean reciprocal rank and success @ K. 

The sample data consists of three csv files:
### 1) trial_fact_check_post_mapping.csv (50 pairs)
This file contains the mapping between fact checks and social media posts.
It has three fields:
- fact_check_id: the id of the fact check in the trial_fact_checks.csv
- post_id: the id of the post in the trial_posts.csv
- pair_lang: the language info about this mapped pair. For example, spa-eng stands for SMP in Spanish and FC in English. 

### 2) trial_fact_checks.csv (50 fact-checks)
This file contains all fact-checks.
It has four fields:

- fact_check_id
- claim - This is the translated text (see below) of the fact-check claim, original text is also contained.
- instances - Instances of the fact-check – a list of timestamps and URLs.
- title - This is the translated text (see below) of the fact-check title

### 3) trial_posts.csv (47 social media posts)
This file contains all social media posts.
It has five fields:

- post_id
- instances - Instances of the fact-check – a list of timestamps and what were the social media platforms.
- ocr - This is a list of texts and their translated versions (see below) of the OCR transcripts based on the images attached to the post.
- verdicts - This is a list of verdicts attached by Meta (e.g., False information)
- text - This is the text and translated text (see below) of the text written by the user.

## Running the code
Clone the git repo of the project
```
git@github.com:sayontang2/claim-retrival.git
```

Create the conda environment with the requirement packages needed to run the code. Assume the name of your virtual environment is ```semeval```
```
conda create --name semeval python=3.11.9
conda activate semeval
pip3 install -r requirements.txt
```

Create ```.env``` file with your `OpenAI` API key
```
OPENAI_API_KEY='your_openai_key'
```

# Experiment Configuration

The experimental configurations in this project are set up within the `main.py` file located in the `/src` folder. The `setup_config()` function handles both hardcoded settings and command-line arguments to define the configuration for each experiment.

## Data Setup

Before running the experiments, ensure that you have the following data folders in your project directory:<br>
**<span style="color: red;"><strong>⚠️</strong> As per the original MultiClaim data owner request, these datasets should be used for research purposes ONLY.</span>**<br>

1. **`/in_data`**  

   Download the Multiclaim Data and place the content (see below) in `in_data` folder.
   The data can be requested at [url-1](https://zenodo.org/records/7737983).  
   This folder contains the training data, with the following components (see the snapshot of their sample):
  - `fact_checks.csv`

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>fact_check_id</th>
      <th>claim</th>
      <th>instances</th>
      <th>title</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>205750</th>
      <td>205750</td>
      <td>('🕋🗃ABRIMOS A CAIXA PRETA DO BNDES.........\n✅...</td>
      <td>[(1574899140.0, 'https://projetocomprova.com.b...</td>
      <td>('Post mistura dados verdadeiros com números i...</td>
    </tr>
  </tbody>
</table>
</div>

   - `posts.csv`

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>post_id</th>
      <th>instances</th>
      <th>ocr</th>
      <th>verdicts</th>
      <th>text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>28088</th>
      <td>28088</td>
      <td>[(1645828783.0, 'fb')]</td>
      <td>[('Voiko rokotteen mRNA jäädä osaksi ihmisen\n...</td>
      <td>['False information']</td>
      <td>('🧪 Tänään 25.2.2022 julkaistu tutkimus osoitt...</td>
    </tr>
  </tbody>
</table>
</div>

   - `fact_check_post_mapping.csv`
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>fact_check_id</th>
      <th>post_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>31301</th>
      <td>205728</td>
      <td>2670</td>
    </tr>
  </tbody>
</table>
</div>

2. **`/sample_data`**  
   Download the `sample_data` folder from [url-2](https://drive.google.com/drive/folders/1pC6YqG4IK8sHGXFUALfnJJBq0nqncbVP?usp=sharing).  
   This folder contains the sample evaluation data, including:
   - `trial_fact_checks.csv`
   - `trial_posts.csv`
   - `trial_data_mapping.csv`

3. **`./openai-op`**  
   Download the `openai-op` folder from [url-3](https://drive.google.com/drive/folders/1PfpXSUp4J4HqkQo6IPa9OMRtvchlJO3u?usp=sharing).  
   This folder contains OpenAI embeddings for of the posts & facts in both training and evaluation data. If you change the fact or the post training (or eval) data, then you would need to get & store the openai (small & large) embeddings for each post/facts as they are neededed for model training and evaluation. The primary key for the post embedding file is ```post_id```, and for facts is ```fact_check_id```.<br>
   Below are the description of the embedding files for the post & facts along with their format (along with a sample) of the embedding files that the codebase assumes.

  - `{eval_}orig-fact.pkl`: External embedding for the facts in the original language.

   <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>gpt_large_emb</th>
      <th>fact_check_id</th>
      <th>gpt_small_emb</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[-0.037123244, 0.032046806000000004, 0.0028377...</td>
      <td>0</td>
      <td>[-0.022125114, 0.020478178, 0.05030176, -0.035...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>[0.0037702448, -0.007277449, -0.01597127, 0.04...</td>
      <td>1</td>
      <td>[-0.0033078135000000003, -0.033126075000000005...</td>
    </tr>
  </tbody>
</table>
</div>

   - `{eval_}eng-fact.pkl`: External embedding for the facts in the english language.

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>gpt_large_emb</th>
      <th>fact_check_id</th>
      <th>gpt_small_emb</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[-0.037099230000000004, 0.032050557, 0.0028363...</td>
      <td>0</td>
      <td>[-0.022196062000000003, 0.020639786, 0.0502345...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>[0.0037702448, -0.007277449, -0.01597127, 0.04...</td>
      <td>1</td>
      <td>[-0.0033078135000000003, -0.033126075000000005...</td>
    </tr>
  </tbody>
</table>
</div>

   - `{eval_}l1-post.pkl`: External embedding for the posts in the original language.

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>gpt_large_emb</th>
      <th>post_id</th>
      <th>gpt_small_emb</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[-0.049027223, 0.0017845220000000002, -0.00299...</td>
      <td>0</td>
      <td>[0.015664421, -0.030023474, -0.006704847000000...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>[0.03373469, -0.024850484000000003, -0.0168357...</td>
      <td>1</td>
      <td>[0.021449845000000002, 0.009177259, 0.03826949...</td>
    </tr>
  </tbody>
</table>
</div>

   - `{eval_}l2-post.pkl`: External embedding for the posts in the english language.

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>gpt_large_emb</th>
      <th>post_id</th>
      <th>gpt_small_emb</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[-0.05174981, -0.026620992000000003, -0.009032...</td>
      <td>0</td>
      <td>[0.010608910000000001, -0.01751335, 0.00185764...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>[0.033766627, -0.024819104, -0.016804272000000...</td>
      <td>1</td>
      <td>[0.021118093, 0.0089649875, 0.03777282, 0.0688...</td>
    </tr>
  </tbody>
</table>
</div>   

## Code Structure

- **`/src/main.py`**: Contains the primary setup for experiments, including hyperparameters, paths, and other configurations. 
- **`setup_config(args_dict)`**: Combines a base config with user-provided settings via command-line arguments.
- **`run_experiment.sh`**: A shell script to automate running multiple experiments with different configurations.

### Configuration Overview

The base configuration is defined within the `setup_config()` function, and it includes the following key settings:

- **Model Hyperparameters**:
  - `feature_type`: Type of features to be used. (e.g., `SBERT_ONLY`)
  - `agg_type`: Aggregation method (e.g., `LINEAR` or `ATTENTION`).
  - `nheads`: Number of attention heads.
  - `model_name`: Pre-trained model name (e.g., `sentence-transformers/use-cmlm-multilingual`).
  - `device`: Device for model training (`cuda:0` or `cpu`).
  - `seed`: Random seed for reproducibility.
  - `train_size`: Percentage of training data to be used.

- **Data Paths**:
  - Paths to input data, including fact checks, posts, and embedding files. These paths can be modified based on the experiment type.

- **Training Hyperparameters**:
  - `loss_scale`, `train_batch_size`, `n_epochs`, `wt_decay`, `lr`: Control training dynamics such as batch size, learning rate, etc.
  - `K`: Number of samples for evaluation or training steps.

- **Logging & Saving**:
  - `log_path`: Path to save experiment logs.
  - `checkpoint_dir`: Directory to save model checkpoints.
  - `model_checkpointing`: Boolean flag to enable/disable checkpoint saving.

### Automating Experiments with `run_experiment.sh`

The `run_experiment.sh` file is a shell script designed to automate the process of running multiple experiments with different configurations. It allows you to loop over combinations of feature sets, aggregation methods, and training data sizes, with additional handling for attention-based models.

#### Script Overview

- **Feature Sets (`feature_types`)**:
  - Defines different types of features to use (e.g., `SBERT_ONLY`, `SBERT_EXT1`).
  
- **Aggregation Methods (`agg_types`)**:
  - Defines the aggregation method, either `LINEAR` (1) or `ATTENTION` (2).

- **Training Data Sizes (`train_sizes`)**:
  - Defines different fractions of the training data to use (e.g., 0.01, 0.05, 0.1).

- **Attention Heads (`nheads_list`)**:
  - Defines the number of attention heads, applicable only when using the `ATTENTION` aggregation method.

#### Example Usage

To run the experiments, simply execute the script:

```bash
./run_experiment.sh
```

This script will:

- Loop over all combinations of `feature_types`, `agg_types`, and `train_sizes`.
- If the `agg_type` is `ATTENTION`, it will iterate over `nheads` values.
- For each configuration, it will call the `main.py` script with appropriate arguments.

Example command generated by the script:

```bash
python src/main.py --feature_type 1 --agg_type 2 --nheads 8 --train_size 0.05
```

### Zero-Shot Evaluation

To run the zero-shot evaluation models, use the provided notebook:

- **Notebook**: `helpers/Zero-shot-eval.ipynb`

This notebook is designed for testing zero-shot models using the data and configuration described above.

### Logging

The `log_path` is dynamically generated based on the configuration, ensuring that logs are stored in experiment-specific directories for easy tracking.

### Experiment Outputs

- Logs are saved in the `./logs/<seed>-<train_size>/` directory with detailed information about the experiment's configuration and progress.
- Model checkpoints are saved in the `./checkpoints/` directory if checkpointing is enabled.

--- 