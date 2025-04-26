import fire
import polars as pl
from tqdm.auto import tqdm
from ranx import Qrels, Run, evaluate
from transformers import AutoTokenizer, AutoModel
from safetensors.torch import save_file
import torch as th


def prepare_embs(
    dataset: str = './med_link_datasets/IfMedLink-name.csv',
    output: str = './exps/name-only/If_SapBERT.safetensors',
    model_name: str = 'cambridgeltl/SapBERT-from-PubMedBERT-fulltext',
):
    # prepare models
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to(device)
    print(f'Using device: {device}. Model: {model_name} loaded.')
    # load dataset
    df = pl.read_csv(dataset)
    corpus = df['med_name_generic'].to_list()
    queries = df['med_name'].to_list()
    # get embs
    bs = 64
    corpus_embs = []
    model.eval()
    for i in tqdm(range(0, len(corpus), bs)):
        batch = corpus[i : i + bs]
        inputs = tokenizer(batch, padding=True, truncation=False, return_tensors="pt").to(device)
        with th.no_grad():
            outputs = model(**inputs)
        corpus_embs.append(outputs.last_hidden_state[:, 0, :].cpu().detach())
    corpus_embs = th.cat(corpus_embs, dim=0)
    queries_embs = []
    for i in tqdm(range(0, len(queries), bs)):
        batch = queries[i : i + bs]
        inputs = tokenizer(batch, padding=True, truncation=False, return_tensors="pt").to(device)
        with th.no_grad():
            outputs = model(**inputs)
        queries_embs.append(outputs.last_hidden_state[:, 0, :].cpu().detach())
    queries_embs = th.cat(queries_embs, dim=0)
    print(f'Corpus embs shape: {corpus_embs.shape}. Queries embs shape: {queries_embs.shape}')
    tensors = {
        "corpus_embs": corpus_embs,
        "queries_embs": queries_embs,
    }
    save_file(
        tensors,
        output,
    )
    print(f'Saved embs to {output}')


if __name__ == "__main__":
    fire.Fire()