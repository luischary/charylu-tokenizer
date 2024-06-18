from pathlib import Path
import json
from multiprocessing import Pool
import time

import pandas as pd
import numpy as np
from tqdm import tqdm
from tokenizers import Tokenizer, pre_tokenizers

from charylutokenizer import CharyluTokenizer


def run_pool_async(num_workers: int, function, payload, chunk_size):
    p = Pool(processes=num_workers)
    result = p.map_async(function, payload, chunk_size)
    num_left = None
    while not result.ready():
        if result._number_left != num_left:
            num_left = result._number_left
            print(f"num left: {num_left}")
        time.sleep(1)

    resultado = result.get()
    p.close()
    return resultado


# amostragem
# 25% textos em ingles
# 25% codigo
# 50% portugues

pesos_bases = {
    "ajibawa_2023_Code_290k_ShareGPT": 50_000,  # 288293
    "alpaca": 0,  # 24548
    "blogset_br": 500_000,  # 4376595
    "cc100": 500_000,  # 24600807
    "commom_crawl": 250_000,  # 4784568
    "cshorten_ml_arxiv_papers": 50_000,  # 117150
    "dkyoon_slimpajama_6b": 50_000,  # 5446162
    "livros": 500,  # 1164
    "open_orca_slimorca_dedup": 50_000,  # 362656
    "pdfs_dominio_publico": 25_000,  # 147165
    "pdfs_tjsp": 150,  # 361
    "sentencas": 500_000,  # 576458
    "itd_stj": 100_000,  # 209135
    "vikp_textbook_quality_programming": 10_000,  # 11650
    "wikipedia_en": 750_000,  # 5268614
    "wikipedia_pt": 700_000,  # 780215
    "the_stack_v1_html": 50_000,  # 3236204
    "the_stack_v1_markdown": 50_000,  # 1735614
    "the_stack_v1_shell": 50_000,  # 1653362
    "the_stack_v1_python": 50_000,  # 781885
    "the_stack_v1_powershell": 50_000,  # 247159
    "the_stack_v1_sql": 15_000,  # 15290
}


def get_tokenizer_texts(
    percent_en: float = 0.20, code: bool = True, quantidade_jur: int = 50_000
):
    """
    No final queremos algo assim
     - pt - 75%
     - en - 20%
     - code - 5%
    """
    escolhidos = []
    pt_len = 0
    # primeiro vamos filtrar duplicidades e linguas que queremos
    df = pd.read_parquet(
        "/media/luis/BIGGER/datasets/nlp_datasets/metadata/all_metadata_nodup_lang.pq"
    )

    df = df[df.duplicado_80 == 0].reset_index(drop=True)
    df = df[df.lang.isin(["pt", "en", "fr", "it", "es"])].reset_index(drop=True)

    if code:
        # pega tudo que for code - pt
        sample = df[(df.tipo == "code") & (df.lang == "pt")].sample(frac=1.0)
        escolhidos += sample.index.tolist()
        pt_len += sample.txt_len.sum()
        print(pt_len)
    else:
        df = df[df.tipo != "code"].reset_index(drop=True)

    # temos uma quantidade minima de juridico que temos que pegar tambem
    if pt_len > 0:
        soma_jur = 0
        for index, row in df[df.tipo == "juridico"].sample(frac=1).iterrows():
            soma_jur += row.txt_len
            escolhidos += [index]

            if soma_jur >= pt_len:
                break
        pt_len += soma_jur
    else:
        sample = df[df.tipo == "juridico"].sample(quantidade_jur)
        escolhidos += sample.index.tolist()
        pt_len += sample.txt_len.sum()
    print(pt_len)

    # segundo tem que pegar todos os pdfs ja que sao a qualidade top
    sample = df[df.tipo == "pdf"]
    # primeiro adiciona as outras linguas
    escolhidos += sample[sample.lang.isin(["fr", "es", "it"])].index.tolist()
    # agora pega os caras de forma a nao desbalancear o que ja tinha
    soma_pdfs = 0
    for index, row in sample[sample.lang == "pt"].sample(frac=1).iterrows():
        soma_pdfs += row.txt_len
        escolhidos += [index]

        if soma_pdfs >= pt_len:
            break
    pt_len += soma_pdfs
    print(pt_len)
    # wikipedia teoricamente eh uma boa fonte tambem
    sample = df[df.base == "wikipedia_pt"].sample(frac=1.0)
    # para nao desbalancear o que ja temos, vamos dobrar o tamanho so
    soma_wiki = 0
    for index, row in sample.iterrows():
        soma_wiki += row.txt_len
        escolhidos += [index]

        if soma_wiki >= pt_len:
            break
    pt_len += soma_wiki
    print(pt_len)
    # denovo para nao desbalancear, vamos acrescentar o conhecimento das ruas (web) tomando cuidado com o len
    soma_web = 0
    sample = df[(df.tipo == "web") & (df.lang == "pt")].sample(frac=1.0)
    for index, row in sample.iterrows():
        soma_web += row.txt_len
        escolhidos += [index]

        if soma_web >= pt_len:
            break
    pt_len += soma_web
    print(pt_len)
    # agora precisamos balancear portugues e ingles, amostrando aleatoriamente
    escolhidos = set(escolhidos)
    # pega as proporcoes
    df_prop = df[df.index.isin(escolhidos)]
    print(df_prop.lang.value_counts(normalize=True))
    escolhidos = list(escolhidos)
    soma_total = df_prop.txt_len.sum()

    prop_en = df_prop[df_prop.lang == "en"].txt_len.sum() / soma_total
    while prop_en < percent_en:
        escolhidos += df[df.lang == "en"].sample(25_000).index.tolist()

        escolhidos = set(escolhidos)
        # pega as proporcoes
        df_prop = df[df.index.isin(escolhidos)]
        print(df_prop.lang.value_counts(normalize=True))
        escolhidos = list(escolhidos)
        soma_total = df_prop.txt_len.sum()

        prop_en = df_prop[df_prop.lang == "en"].txt_len.sum() / soma_total
        print(prop_en)

    escolhidos = set(escolhidos)
    df_prop = df[df.index.isin(escolhidos)].reset_index(drop=True)
    print(df_prop)
    return df_prop


def validate_distribution():
    df = pd.read_parquet(
        "/media/luis/BIGGER/datasets/nlp_datasets/metadata/all_metadata_nodup.pq"
    )

    df = df[df.duplicado_80 == 0].reset_index(drop=True)
    escolhidos = []
    df["txt_len"] = df["txt_len"].astype(int)

    for base in tqdm(df.base.unique()):
        print(base)
        df_base = df[df.base == base]
        amostra = df_base.sample(pesos_bases[base])
        escolhidos.append(amostra)

    escolhidos = pd.concat(escolhidos).reset_index(drop=True)
    soma_texto = escolhidos.txt_len.sum()
    print(escolhidos)
    print(escolhidos.groupby("base").txt_len.sum() / soma_texto)
    print(escolhidos.groupby("lang").txt_len.sum() / soma_texto)
    return escolhidos


def iterate_over_texts(df: pd.DataFrame = None):
    if df is None:
        df = pd.read_parquet(
            "/media/luis/BIGGER/datasets/nlp_datasets/metadata/all_metadata_nodup.pq"
        )
        df = df[df.duplicado_80 == 0].reset_index(drop=True)

    df = df.sort_values("partition_ref").reset_index(drop=True)

    particao_atual = None
    nome_particao_atual = None
    for _, row in df.iterrows():
        particao = row.partition_ref
        linha_particao = int(row.partition_index)

        if particao_atual is None or nome_particao_atual != particao:
            # precisa recarregar a particao
            particao_atual = pd.read_parquet(particao)
            nome_particao_atual = particao

        texto = particao_atual.loc[linha_particao, "text"]
        yield texto


def pega_textos_particao(dados):
    partition_path, partition_indexes = dados
    print(partition_path)

    particao = pd.read_parquet(partition_path)
    textos = particao[particao.index.isin(partition_indexes)].text.tolist()
    return textos


def get_textos_tokenizacao(df: pd.DataFrame):
    payloads = []
    for particao in tqdm(df.partition_ref.unique()):
        base = df[df.partition_ref == particao]
        indexes = base.partition_index.tolist()

        payloads.append([particao, indexes])

        if len(payloads) % 160 == 0 and len(payloads) > 0:
            lista_textos = run_pool_async(
                num_workers=32,
                function=pega_textos_particao,
                payload=payloads,
                chunk_size=1,
            )
            for l in lista_textos:
                for t in l:
                    yield t
            payloads = []

    lista_textos = run_pool_async(
        num_workers=32, function=pega_textos_particao, payload=payloads, chunk_size=1
    )
    for l in lista_textos:
        for t in l:
            yield t


def shrink_tokenizer(base_tokenizer_path, vocab_keep_items, save_path):
    # codigo https://discuss.huggingface.co/t/tokenizer-shrinking-recipes/8564
    tokenizer = CharyluTokenizer(tokenizer_path=base_tokenizer_path).tokenizer
    # assert tokenizer.is_fast, "This only works for fast tokenizers."
    # tokenizer_json = json.loads(tokenizer._tokenizer.to_str())
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer_json = json.loads(tokenizer.to_str())
    vocab = tokenizer_json["model"]["vocab"]
    if tokenizer_json["model"]["type"] == "BPE":
        new_vocab = {token: i for token, i in vocab.items() if i < vocab_keep_items}
        merges = tokenizer_json["model"]["merges"]
        new_merges = []
        for i in range(len(merges)):
            # a, b = merges[i].split()
            a, b = merges[i].split(" ")
            new_token = "".join((a, b))
            if a in new_vocab and b in new_vocab and new_token in new_vocab:
                new_merges.append(merges[i])
        tokenizer_json["model"]["merges"] = new_merges
    elif tokenizer_json["model"]["type"] == "Unigram":
        new_vocab = vocab[:vocab_keep_items]
    elif (
        tokenizer_json["model"]["type"] == "WordPiece"
        or tokenizer_json["model"]["type"] == "WordLevel"
    ):
        new_vocab = {token: i for token, i in vocab.items() if i < vocab_keep_items}
    else:
        raise ValueError(f"don't know how to handle {tokenizer_json['model']['type']}")
    tokenizer_json["model"]["vocab"] = new_vocab
    # tokenizer._tokenizer = Tokenizer.from_str(json.dumps(tokenizer_json))
    # tokenizer.save_pretrained(save_path)
    tokenizer = Tokenizer.from_str(json.dumps(tokenizer_json))
    tokenizer.save(save_path)


def simple_iterator(texts):
    for t in texts:
        yield t


def file_text_generator(file_path):
    texto_documento = ""
    with open(file_path, "r", encoding="utf8") as f:
        for texto_linha in f:
            if texto_linha.find("<END_OF_DOC>") < 0:
                texto_documento += "\n" + texto_linha
            else:
                if np.random.random() >= 0:
                    yield texto_documento
                texto_documento = ""


if __name__ == "__main__":
    dados = get_tokenizer_texts(
        code=False, quantidade_jur=10_000, percent_en=0.15
    )  # .iloc[:10_000]
    textos = []
    for t in get_textos_tokenizacao(dados):
        textos.append(t)
    # salva os dados
    Path(
        "/home/luis/projetos/luis_transformers/tokenizer/training_data.txt"
    ).write_text("\n<END_OF_DOC>\n".join(textos))
    textos = None
    del textos

    for vocab_size, vocab_name in zip(
        [150000],
        ["150k"],
    ):
        novo_tokenizer = CharyluTokenizer(
            vocab_size=vocab_size,
            tokenizer_path=f"/home/luis/projetos/luis_transformers/artifacts/charylu_nocode/tokenizer_2024_{vocab_name}.json",
        )
        novo_tokenizer.train(
            file_text_generator(
                "/home/luis/projetos/luis_transformers/tokenizer/training_data.txt"
            )
        )
        novo_tokenizer = None

    # para os demais vamos manipular e remover os tokens amais
    for vocab_size, vocab_name in zip(
        [32000, 50000, 60000, 70000, 80000, 90000, 100000, 110000, 120000, 130000],
        ["32k", "50k", "60k", "70k", "80k", "90k", "100k", "110k", "120k", "130k"],
    ):
        shrink_tokenizer(
            base_tokenizer_path="/home/luis/projetos/luis_transformers/artifacts/charylu_nocode/tokenizer_2024_150k.json",
            vocab_keep_items=vocab_size,
            save_path=f"/home/luis/projetos/luis_transformers/artifacts/charylu_nocode/tokenizer_2024_{vocab_name}.json",
        )
