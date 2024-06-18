from charylutokenizer.charylutokenizer import CharyluTokenizer
from charylutokenizer.load import load


def test_tokenizer():
    tokenizer = CharyluTokenizer(
        tokenizer_path="charylutokenizer/artifacts/charylu_nocode/tokenizer_2024_90k.json"
    )
    tokens = tokenizer.tokenize("Texto para teste 123")
    print(tokens)
    assert type(tokens) == list
    assert len(tokens) > 0
    resposta = [48137, 9973, 16941, 287, 62, 63, 64]
    for i in range(len(resposta)):
        assert resposta[i] == tokens[i]


def test_load():
    tokenizer = load(90, "_nocode")
    tokens = tokenizer.tokenize("Texto para teste 123")
    print(tokens)
    assert type(tokens) == list
    assert len(tokens) > 0
    resposta = [48137, 9973, 16941, 287, 62, 63, 64]
    for i in range(len(resposta)):
        assert resposta[i] == tokens[i]
