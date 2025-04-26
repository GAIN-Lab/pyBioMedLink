# edit distance
uv run python baselines/edit_dist.py run
uv run python baselines/edit_dist.py run --dataset './med_link_datasets/NIfMedLink-name.csv' --output './exps/name-only/edit_dist_NIf_name_result.csv'

# BM25
uv run python baselines/bm25.py run
uv run python baselines/bm25.py run  --dataset './med_link_datasets/NIfMedLink-name.csv' --output './exps/name-only/bm25_NIf_name_result.csv'

# SapBERT
## Step1: prepare embedding in advance
 uv run python baselines/bert_emb.py prepare_embs