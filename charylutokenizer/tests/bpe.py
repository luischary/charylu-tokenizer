from pathlib import Path
import re
from typing import List
import json
import time
from multiprocessing import Pool
import sys

from tqdm import tqdm

from util.mytimeit import timeit
from itertools import pairwise
from datastructures.multiset import Multiset
from datastructures.indexlist import IndexedList

re_pretokenize = r"\d|[^\w\s]{3}|[^\w\s]{1}| ?[A-Z][^\d\s\WA-Z]+| ?[A-Z]?[^\d\s\W]+|\s{16}|\s{12}|\s{8}|\s{4}|\s"


class CharyluTokenizer:
    """
    Meu proprio BPE tokenizer
    """

    def __init__(self, tokenizer_path: str = None, vocab_size: int = None) -> None:
        self.vocab_size = vocab_size
        self.tokenizer_path = tokenizer_path
        self.loaded = False
        self.vocab = None
        self.merges = None
        self.bytes_vocab = None
        self.pool = Pool(processes=32)

        if tokenizer_path is not None:
            if Path(self.tokenizer_path).exists():
                self.load()
                self.loaded = True

    def _portuguese_split(self, text):
        re_suffix = r"escer\b|mente\b|alhão\b|mento\b|iscar\b|ência\b|douro\b|tório\b|tério\b|ismo\b|tica\b|agem\b|ário\b|eiro\b|aria\b|isco\b|eiro\b|ista\b|dade\b|eira\b|inho\b|acho\b|icho\b|ento\b|aréu\b|arra\b|into\b|enho\b|onho\b|iaco\b|ismo\b|aico\b|arrão\b|eirão\b|ense\b|ista\b|ável\b|ível\b|óvel\b|úvel\b|agem\b|ança\b|eria\b|ença\b|ejar\b|icar\b|itar\b|ecer\b|ano\b|ena\b|eno\b|ino\b|iço\b|ivo\b|ude\b|dor\b|iam\b|ndo\b|aço\b|uça\b|tor\b|eco\b|ela\b|ote\b|sor\b|ume\b|nte\b|rio\b|ura\b|eza\b|ice\b|ado\b|ato\b|ada\b|ama\b|ica\b|ame\b|edo\b|oso\b|udo\b|aco\b|ata\b|ção\b|ear\b|az\b|ez\b|am\b|ão\b|eo\b|ão\b|or\b|ia\b|al\b|ês\b|eu\b|s\b"
        re_preffix = r" ?\bambi| ?\banti| ?\bante| ?\bbem| ?\bben| ?\bbi| ?\bbis| ?\bcontra| ?\bin| ?\bi| ?\bpos| ?\bsemi| ?\btri"
        splits = []
        splits_pre = re.findall(re_preffix, text, flags=re.IGNORECASE)
        if len(splits_pre) > 0:
            # pega a primeira que encontrar
            achou = splits_pre[0]
            splits.append(achou)
            tamanho_achou = len(achou)
            # remove do text
            text = text[tamanho_achou:]
        # agora o suffixo
        splits_pos = re.findall(re_suffix, text)
        if len(splits_pos) > 0:
            achou = splits_pos[0]
            tamanho_achou = len(achou)
            text = text[:-tamanho_achou]
            if len(text) > 0:
                splits.append(text)
            splits.append(achou)
        else:
            splits.append(text)

        return splits

    def _preprocess_text(self, text):
        text = re.sub(r"\n(.)\n", "\n", text)
        text = re.sub(r"[\n\n]+", "\n\n", text)
        return text

    def _pre_tokenize(self, text: str) -> List[int]:
        """
        Faz a pre-tokenizacao e depois ja converte para bytes
        """
        pre_split = re.findall(
            re_pretokenize, self._preprocess_text(text), flags=re.IGNORECASE
        )
        # agora aplica o prefixo e sufixo do portugues
        # p_split = self._portuguese_split(next(pre_split_iterator))
        # for s in pre_split_iterator:
        #     p_split += self._portuguese_split(s)
        return pre_split

        if len(pre_split) > 0:
            b_splits = list(pre_split.pop(0).encode("utf8"))
            if len(pre_split) > 0:
                for t in pre_split:
                    b_splits += [-1] + list(t.encode("utf8"))
        else:
            b_splits = []
        return b_splits

    def _pre_tokenize_from_iterator(self, text_iterator) -> List[int]:
        pre_split = []
        for text in text_iterator:
            pre_split += re.findall(
                re_pretokenize, self._preprocess_text(text), flags=re.IGNORECASE
            )
        # agora aplica o prefixo e sufixo do portugues
        # p_split = self._portuguese_split(next(pre_split_iterator))
        # for s in pre_split_iterator:
        #     p_split += self._portuguese_split(s)

        if len(pre_split) > 0:
            b_splits = list(pre_split.pop(0).encode("utf8"))
            if len(pre_split) > 0:
                for t in pre_split:
                    b_splits += [-1] + list(t.encode("utf8"))
        else:
            b_splits = []
        return b_splits

    def build_indexed_list(
        self, tokens: List[int]
    ):  # Create an IndexedList with the encoded bytes.
        return IndexedList(t for t in tokens)

    def init_pairs_stats(
        self, tokens: List[int]
    ):  # Initialize a Multiset with all overlapping pairs.
        # For text "aaabd" the multiset will contain: {(a,a): 2, (a, b): 1, (b, d): 1}
        return Multiset(pairwise(t for t in tokens))

    def build_everything(self, text_iterator):
        index_list = None
        multiset = Multiset()
        for text in tqdm(text_iterator):
            pre_tokenized = self._pre_tokenize(text)

            if index_list is None:
                index_list = IndexedList(t for t in pre_tokenized)
            else:
                pre_tokenized = [-1] + pre_tokenized
                index_list.add(t for t in pre_tokenized)

            pares = pairwise(t for t in pre_tokenized)
            for p in pares:
                multiset.add(p)
            multiset._commit()
        return index_list, multiset

    def merge(self, pair, new_id, indexed_list: IndexedList, stats: Multiset = None):
        if pair in indexed_list.stale_index:
            for node in indexed_list.stale_index[pair]:
                if node.val != pair[0] or node.next is None or node.next.val != pair[1]:
                    continue  # The index was stale - continue.
                # Say we're merging "bc" to "X" in "abcd", and the node we're visiting now is "b".
                if stats is not None:  # Update the stats.
                    stats.remove(pair)  # Remove "bc".
                    # to_remove.append(pair)
                    if node.next.next is not None:
                        stats.remove(
                            (node.next.val, node.next.next.val)
                        )  # Remove "cd".
                        # to_remove.append((node.next.val, node.next.next.val))
                        stats.add((new_id, node.next.next.val))  # Add "Xd".
                        # to_add.append((new_id, node.next.next.val))
                    if node.prev is not None:
                        stats.remove((node.prev.val, pair[0]))  # Remove "ab".
                        # to_remove.append((node.prev.val, pair[0]))
                        stats.add((node.prev.val, new_id))  # Add "aX".
                        # to_add.append((node.prev.val, new_id))
                node.next.delete()  # Delete "c", we now have "abd".
                node.val = new_id  # Update "b" to "X", we now have "aXd".
                indexed_list.update_index(node)  # Add "aX" and "Xd" to the index.

    def train(self, texts, vocab_size, verbose=False):
        print("PRE_PROCESSANDO")
        # tokens = []
        stats = Multiset()
        indexed_list = None
        for idx, t in tqdm(enumerate(texts)):
            pre_tokenized = self._pre_tokenize(t)

            for t in pre_tokenized:
                if indexed_list is None:
                    indexed_list = IndexedList(i for i in t.encode("utf8"))
                else:
                    indexed_list.add(i for i in t.encode("utf8"))
                for p in pairwise(i for i in t.encode("utf8")):
                    stats.add(p)

            if idx % 10_000 == 0:
                stats._commit()
                print(
                    sys.getsizeof(t) / 1024**3,
                    sys.getsizeof(pre_tokenized) / 1024**3,
                    sys.getsizeof(indexed_list) / 1024**3,
                    indexed_list.count(),
                    sys.getsizeof(stats) / 1024**3,
                    stats.count(),
                )

        # print(
        #     f"Training tokenizer on text of length {len(tokens):,} with vocab of size {vocab_size:,}."
        # )

        n_merges = vocab_size - 256
        vocab = {i: bytes([i]) for i in range(256)}
        merge_tree = []
        # indexed_list = timeit(
        #     lambda: self.build_indexed_list(tokens), "build_indexed_list"
        # )
        # stats = timeit(lambda: self.init_pairs_stats(tokens), "init_pairs_stats")
        # tokens = None
        # del tokens
        # indexed_list, stats = self.build_everything(texts)
        for i in tqdm(range(n_merges)):
            if not stats:
                break  # Stop if we don't have any pairs (we should probably stop earlier).

            par_valido = False
            while not par_valido:
                top_pair = stats.most_common
                if -1 in top_pair:
                    stats.remove(top_pair, stats.d[top_pair].count)
                    continue
                else:
                    par_valido = True

            new_id = len(vocab)
            merge_tree.append((top_pair, new_id))
            vocab[new_id] = vocab[top_pair[0]] + vocab[top_pair[1]]
            if verbose:
                print(
                    f"Merge {i+1}/{n_merges}: {top_pair} -> {new_id} ({vocab[new_id]}) had {stats.count(top_pair)} occurrences"
                )
            self.merge(top_pair, new_id, indexed_list, stats)

        self.merges = merge_tree
        self.bytes_vocab = vocab
        self.save()

    def tokenize(self, text):
        indexed_list = None
        pre_tokenized = self._pre_tokenize(text)
        tokens = []
        for t in pre_tokenized:
            tokens += list(t.encode("utf8")) + [-1]

        indexed_list = IndexedList(i for i in tokens)

        for pair, new_id in self.merges:
            self.merge(pair, new_id, indexed_list, None)

        tokenized = [node.val for node in indexed_list if node.val != -1]
        return tokenized

    def detokenize(self, seq):
        return b"".join((self.bytes_vocab[t] for t in seq)).decode(
            "utf-8", errors="replace"
        )

    def save(self):
        obj = {
            "vocab": {
                self.detokenize([idx]): idx for idx in range(len(self.bytes_vocab))
            },
            "merges": [list(m[0]) for m in self.merges],
        }

        obj_json = json.dumps(obj)
        folder = Path(self.tokenizer_path)
        folder.mkdir(parents=True, exist_ok=True)
        file = folder / "tokenizer.json"
        file.write_text(obj_json, encoding="utf8")

    def load(self):
        arquivo = Path(self.tokenizer_path) / "tokenizer.json"
        texto = arquivo.read_text(encoding="utf8")
        dados = json.loads(texto)
        tokens_vocab = [i for i in range(256)]
        self.bytes_vocab = {i: bytes([i]) for i in range(256)}
        meus_merges = []
        merges = dados["merges"]
        for idx, par in enumerate(merges):
            m = tuple(par)
            meus_merges.append((m, 256 + idx))
            tokens_vocab.append(256 + idx)
            self.bytes_vocab[idx + 256] = (
                self.bytes_vocab[m[0]] + self.bytes_vocab[m[1]]
            )

        self.vocab = tokens_vocab
        self.merges = meus_merges
        self.vocab_size = len(self.vocab)


if __name__ == "__main__":
    textos = [
        "paralelepipedo uma Iniciativa da literatura bemmelhante! huhu",
        "uma nova historia emocionante da literatura mundial",
        "o rato roeu a roupa do rei de roma aquele arrombadinho do caralho",
        "eu gósto de falar com açentos malúcos enquânto isso!",
        """
0

Login
Guia Enem
Língua Portuguesa
Matemática
Geografia
História
Física
Química
Biologia
Inglês
Todas as Disciplinas 
BOLSAS DE ESTUDOGUIA ENEMLÍNGUA PORTUGUESAPREFIXO E SUFIXO
PREFIXO E SUFIXO
Postado por Amanda Maria Azevedo em 10/07/2019 e atualizado pela última vez em 21/07/2020
Afixos que transformam palavras quando adicionados na frente ou no final de um radical

O prefixo e o sufixo são morfemas da língua portuguesa, também chamadas de afixos, que são usados com radicais de palavras para formar uma nova palavra que passa a ter um novo significado. 

O prefixo deve ser colocado na frente do radical e o sufixo deve ser colocado no final do radical. 

Confira abaixo alguns exemplos: 

- Contradizer (prefixo: contra)
- Antivírus (prefixo: anti)
- Hipertensão (prefixo: hiper)
- Decrescer (sufixo: escer) 
- Trovejar (sufixo: ejar)
- Pugilismo (sufixo: ismo)


O prefixo e o sufixo são muito importantes para a compreensão da função das palavras. (Foto: Shutterstock)

Prefixos 

O prefixo é um tipo de morfema e afixo que forma uma palavra através de um determinado radical. Ele é adicionado na frente do radical, formando uma nova palavra que pode ter um significado diferente, mas continua mantendo a mesma classe gramatical. 

Os prefixos da língua portuguesa são de origem latina ou grega. Veja nas listas abaixo exemplos de prefixos e seus significados: 

Prefixos latinos:
Prefixos	Significados	Exemplos
ab-	afastamento	abdicar
ambi-	duplicação	ambidestro
ante-	anterioridade	antepor
bem-, ben-	bem	bendito, beneficente
bi-, bis-	dois	biênio, bisneto
contra-	oposição	contradizer
in-, i-	negação	ingrato, ilegal
pos-	posição	posterior
semi-	metade	semicírculo
tri-	três	triângulo

Prefixos gregos: 
Prefixos	Significados	Exemplos
anti-	oposição	antipatia
arce-	superioridade	arcebispo
cata-	movimento para baixo	cataclismo
dis-	dificuldade	dispneia
en-	posição interior	encéfalo
epi-	posterioridade	epílogo
eu-	bem, bom	eufonia
hiper-	excessivo	hipertensão
para-	proximidade	paralelo
pro-	anterioridade	prólogo

Sufixos 

O prefixo é um tipo de morfema e afixo que, assim como o prefixo, forma uma palavra através de um determinado radical. Ele é adicionado no final do radical, formando uma nova palavra que pode ter um significado diferente e pode também fazer parte de uma nova classe gramatical. 

Os sufixos podem ser classificados em nominais, verbais e adverbiais.

Confiras nas tabelas abaixo exemplos e tipos de sufixos, junto com seus significados: 
Sufixos Nominais	Sufixos	Exemplos
Sufixos Aumentativos	
-ão
-aço
-alhão
-aréu
-arra
-(z)arrão
-eirão
-uça
paredão
ricaço
grandalhão
povaréu
bocarra
homenzarrão
boqueirão
dentuça
Sufixos Diminutivos	
-inho
-zinho
-acho
-icho (a)
-eco
-ela
-ote
-isco
Pedrinho
avozinho
riacho
barbicha
soneca
viela
velhote
chuvisco

Sufixos nominais:
Sufixos	Exemplos	Significado
-dor
-tor
-sor
-eiro
-ista
-nte
-rio

causador
tradutor
professor
padeiro
dentista
estudante
bibliotecário
agente, profissão, instrumento

-dade
-ência
-ez
-eza
-ice
-ície
-ismo
-or
-ude
-ume
-ura
credibilidade
paciência
sensatez
beleza
meiguice
imundície
patriotismo
frescor
amplitude
azedume
formosura
qualidade, estado
-ado
-ato
-aria
-douro
-tório
-tério
principado
orfanato
padaria
matadouro
dormitório
cemitério
lugar, ramo de negócio
-ia
-ismo
-ica
-tica
geometria
cristianismo
física
política
ciência, técnica, doutrina
-al
-agem
-ada
-ama
-ame
-ário
-aria
-edo
-eiro
-eira
-ena
cafezal
ferragem
boiada
dinheirama
vasilhame
mobiliário
gritaria
arvoredo
formigueiro
fumaceira
dezena

coletivo
-az
-ento
-lento
-into
-enho
-onho
-oso
-udo
sagaz
ciumento
sonolento
faminto
ferrenho
medonho
jeitoso
barrigudo
qualidade em abundância, intensidade
-eo
-aco
-iaco
-aco
-aico
-ano
-ão
-enho
-eno
-ense
-ês
-eu
-ino
-ista
ósseo
demoníaco
paradisíaco
polaco
hebraico
paraibano
catalão
panamenho
chileno
cearense
francês
europeu
argentino
paulista
natureza, origem, que tem a qualidade de
-ável
-ível
-óvel
-úvel
-iço
-ivo

amável
audível
móvel
solúvel
movediço
lucrativo
possibilidade, tendência
-ada
-agem
-ança
-aria
-eria
-ata
-ção
-ura
-ela
-ença
-ência
-mento
-or
cabeçada
aprendizagem
esperança
pirataria
selvageria
passeata
correção
formatura
olhadela
parecença
continência
juramento
temor
ação, resultado de ação

O sufixo verbal quando colocado junto a um determinado radical, transforma a palavra em verbo. Confira abaixo os exemplos:
Sufixos	Exemplos	Significado
-ear
-ejar
folhear, espernear
gotejar, apedrejar
ação que se repete
-icar
-itar
-iscar
bebericar
saltitar
petiscar
ação diminutiva que se repete
-ecer
-escer
amanhecer, anoitecer
florescer, rejuvenescer
ação que principia

O sufixo adverbial é adicional a um radical para se transformar em um advérbio, que é sempre terminado com o sufixo –mente. 

Confira abaixo alguns exemplos: 

- Calmamente 
- Agitadamente
- Tranquilamente
- Antigamente
- Possivelmente
- Intermitente 
- Realmente 
Artigos Relacionados
Flexão dos Adjetivos
A flexão dos adjetivos indica três diferentes maneiras de caracterizar ou qualificar um substantivo: em gênero, número ou grau.

Figuras de Som
As figuras de som estão atreladas à sonoridade das palavras. Classificam-se em: aliteração, assonância, onomatopeia e paranomásia.

Figuras de Palavras
As figuras de palavras são classificadas em: metáfora, metonímia, comparação, perífrase, sinestesia, sinédoque, alegoria e catacrese.

Quer estudar
pagando menos?
Com o Educa Mais Brasil você estuda com desconto até o final do curso!
O que deseja estudar?
Graduação
Selecione o curso que deseja estudar:
Curso que deseja estudar
Fale com o
Educa Mais Brasil
WhatsApp
Seg a Sex: 07h às 20h40
Sábado: 08h às 18h
Siga-nos nas
redes sociais

Facebook
Twitter
Youtube
Instagram
Linkedin
TikTok
Educação
Bolsas de estudos para Faculdades
Cursos de Faculdades
Cursos Técnicos
Escolas
Séries de Educação Básica
Proposta Pedagógica
Graduação
Pós-Graduação
Educação Básica
Cursos Técnicos
Idiomas
Cursos Livres
Pré-ENEM
Preparatório para Concursos
EJA
E+B Educação
Carreira
ENEM
Fies
Notícias
Prouni
Sisu
Dicas
Escolas
Guia Enem
Cursos Gratuitos
Gabarito Enem
Programas do Governo
Notas de Corte
ENEM
Sisu
Prouni
Fies
Pronatec
Sisutec
ENCCEJA
ENARE
Dicas E+B
Teste Vocacional
Teste de Inglês
Cronograma Enem
Redação Enem
O Educa Mais Brasil
Quem Somos
Políticas de Privacidade
Termos de Uso
Como Funciona
Contato
Fale Conosco
Indique um amigo
Assessoria de Imprensa
Nós te ligamos
Whatsapp
Instituição
Portal do Parceiro
Quero ser Parceiro
Prêmio Reclame Aqui
© 2024  Educa Mais Brasil. Todos os direitos reservados
Utilizamos cookies.  Política de Privacidade.
""",
    ]
    tokenizer = CharyluTokenizer(tokenizer_path="artifacts/charylu")
    # tokenizer.train(textos, vocab_size=1000)
    tokens = tokenizer.tokenize(
        "ho ho ho o o o o feliz natal do caralhão Amarelado da porra"
    )
    tokens = [(t, tokenizer.detokenize([t])) for t in tokens]
    print(tokens)
    # # texto = tokenizer.decode(tokens)
    # # print(texto)
    # print(tokenizer.vocab_size)
