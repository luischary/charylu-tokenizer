from pathlib import Path
from typing import List

from tokenizers import (
    decoders,
    models,
    pre_tokenizers,
    trainers,
    normalizers,
    Tokenizer,
)
from transformers import PreTrainedTokenizerFast

from portuguese import PortuguesePreTokenizer


class MyTokenizer:
    def __init__(self, vocab_size: int = 60000, tokenizer_path: str = None) -> None:
        self.vocab_size = vocab_size
        self.tokenizer_path = tokenizer_path
        self.loaded = False

        if Path(self.tokenizer_path).exists():
            self.load_tokenizer()
            self.loaded = True

    def train(self, text_iterator):
        # cria o tokenizer
        tokenizer = Tokenizer(models.BPE())

        # cria o pre-tokenizer
        # ele vai gerar um upper-bound nos tokens
        # queremos que os digitos (numeros) fiquem separados para maior assertividade com numeros
        # tambem queremos que as pontuacoes sejam isoladas para facilitar a utilizacao dos modelos
        # tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
        #     [
        #         pre_tokenizers.Digits(individual_digits=True),
        #         pre_tokenizers.Punctuation(behavior="isolated"),
        #         pre_tokenizers.ByteLevel(add_prefix_space=False),
        #     ]
        # )

        tokenizer.pre_tokenizer = pre_tokenizers.PreTokenizer.custom(
            PortuguesePreTokenizer()
        )

        # essa opcao de add_prefix_space serve para adicionar um espaco no comeco.
        # como eh falsa, nao vamos adicionar um espaco no comeco da string.

        # teste do pre-tokenizer
        print(
            tokenizer.pre_tokenizer.pre_tokenize_str(
                "uma Iniciativa da literatura bemmelhante! huhu"
            )
        )

        # bota um normalizer para substituir algumas coisas
        tokenizer.normalizer = normalizers.Sequence(
            [
                normalizers.Replace("\n(.)\n", "\n"),
                normalizers.Replace("[\n\n]+", "\n\n"),
                normalizers.NFKC(),
            ]
        )

        # cria o trainer do tokenizer
        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=[
                "<pad>",
                "<s>",
                "</s>",
                "<mask>",
                "<cls>",
                "<unk>",
                "<sys>",
                "</sys>",
                "<ctx>",
                "</ctx>",
                "<iauser>",
                "</iauser>",
                "<iaagent>",
                "</iaagent>",
                "<memory>",
            ],
            max_token_length=17,  # para nao criar tokens muito nada a ver. Peguei do calculo de outlier simples em cima da base
        )
        tokenizer.train_from_iterator(text_iterator, trainer=trainer)

        # agora, para nao ficar com aquele G que representa o espaco no comeco da palavra quando
        # detokenizarmos o texto
        # para isso colocarmos um decoder de bytelevel
        # tokenizer.decoder = decoders.ByteLevel()

        # # antes de salvar tira o cara custom que ele nao gosta
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        tokenizer.save(self.tokenizer_path)

        # no final temos que salvar ele de alguma forma
        # vamos usar a classe basica de PreTrainedTokenizerFast
        # wrapped_tokenizer = PreTrainedTokenizerFast(
        #     tokenizer_object=tokenizer,
        #     bos_token="<s>",
        #     eos_token="</s>",
        #     unk_token="<unk>",
        #     pad_token="<pad>",
        #     mask_token="<mask>",
        #     cls_token="<cls>",
        # )

        # wrapped_tokenizer.save_pretrained(self.tokenizer_path)
        # self.tokenizer = wrapped_tokenizer

    def load_tokenizer(self) -> None:
        # self.tokenizer = PreTrainedTokenizerFast.from_pretrained(self.tokenizer_path)
        self.tokenizer = Tokenizer(models.BPE()).from_file(self.tokenizer_path)
        self.tokenizer.pre_tokenizer = pre_tokenizers.PreTokenizer.custom(
            PortuguesePreTokenizer()
        )

    def tokenize_text(
        self, text, padding="do_not_pad", truncation=None, max_length=None
    ) -> List[int]:
        tokenized = self.tokenizer.encode(
            text, padding=padding, truncation=truncation, max_length=max_length
        )
        return tokenized

    def untokenize_tokens(
        self, tokens: List[int], skip_special_tokens: bool = True
    ) -> str:
        return self.tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)
