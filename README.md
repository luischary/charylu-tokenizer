# CharyluTokenizer

Biblioteca elaborada e utilizada por Luis Felipe Chary para utilização em projetos de DeepLearning envolvendo linguagem natural.

Consiste em uma gama de tokenizadores treinados utilizando a biblioteca [Tokenizers](https://github.com/huggingface/tokenizers/tree/main) mas com conjunto de dados próprios. Todos os tokenizadores foram treinados utilizando o algoritmo *byte-pair encoding*.

Os tokenizadores foram especialmente projetados para utilização em linguas latinas, com foco no Português.

## Utilização básica

Para utilizar os tokenizadores utilize o seguinte código de exemplo:
```
from charylutokenizer.load import load

# inicializa o tokenizer
tokenizer = load(90, "_nocode") # tamanho do vocabulario desejado (k) e tipo

# tokeniza string
tokenizer.tokenize("texto de teste") # [47941, 9851, 16941]

# detokeniza tokens
tokenizer.detokenize([47941, 9851, 16941]) # "testo de teste"
```

## Escopo

Os tokenizadores foram treinados em uma base proprietária que possui uma ampla gama de origens de textos (internet, livros, publicações científicas, repositórios de código, etc).

Para o treinamento dos tokenizadores, foi feita uma filtragem principalmente nas línguas de modo a aumentar a eficiência nas linguas latinas e inglês, dessa forma o tokenizador deve comportar bem:

- **Português (Brasil e Portugal)** ~ 80% da base de treino
- **Espanhol** ~ 1.5% da base de treino
- **Italiano** ~ 1.5% da base de treino
- **Francês** ~ 2% da base de treino
- **Inglês** ~ 15% da base de treino


## Versões

Foram treinados dois tipos diferentes de tokenizadores:

1. **Proposta geral** - treinado em textos de múltiplas naturezas.
2. **No code** - treinado em textos de múltiplas naturezas menos programação. Foram removidos textos de repositórios, livros, pdfs... tudo que possuia alguma linguagem de programação.

| Vocab. | Geral | NoCode |
|:------:|:-----:|:------:|
|   32   |   ✓   |    ✓   |
|   50   |   ✓   |    ✓   |
|   60   |   ✓   |    ✓   |
|   70   |   ✓   |    ✓   |
|   80   |   ✓   |    ✓   |
|   90   |   ✓   |    ✓   |
|   100  |   ✓   |    ✓   |
|   110  |       |    ✓   |
|   120  |   ✓   |    ✓   |
|   130  |       |    ✓   |
|   150  |   ✓   |    ✓   |



## Changelog
- version 0.0.5 - first usable version