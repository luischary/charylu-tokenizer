from pathlib import Path
import re
from typing import List
import json
import heapq
from multiprocessing import Pool
import time

from tqdm import tqdm

re_pretokenize = r"\d|[^\w\s]{3}|[^\w\s]{1}| ?[^\d\s\W]+|\s{16}|\s{12}|\s{8}|\s{4}|\s"


def run_pool_async(p, function, payload, chunk_size):
    result = p.map_async(function, payload, chunk_size)
    num_left = None
    quantidade_inicial = len(payload) // chunk_size
    while not result.ready():
        if result._number_left != num_left:
            num_left = result._number_left
            print(
                f"{quantidade_inicial - num_left}/{quantidade_inicial} - {100 * (quantidade_inicial - num_left)/quantidade_inicial:.2f}%"
            )
        time.sleep(0.1)

    resultado = result.get()
    return resultado


def _filtra_resultados(resultados_merge):
    return resultados_merge[0]


def _count_pairs(tokens: List[int]):
    counts = {}
    for par in zip(tokens, tokens[1:]):
        tuplado = tuple(par)
        counts[tuplado] = counts.get(tuplado, 0) + 1

    return counts


def _merge_single(all_inputs):
    respostas = []
    for input in all_inputs:
        respostas.append(_merge(input))
    return respostas


def _merge(input):
    """
    pair: tuple, new_token, tokens: List[int]
    """
    pair, new_token, tokens = input
    # converte tudo para texto para fazer o merge mais facil
    # pair_txt = "_".join(str(p) for p in pair)
    # tokens_txt = "_".join(str(t) for t in tokens)
    # merged = re.sub(pair_txt, str(new_token), tokens_txt)
    # novos = [int(i) for i in merged.split("_")]
    novos = []
    i = 0
    contagem_que_importa = {}
    contagem_teste = _count_pairs(tokens)
    if pair not in contagem_teste:
        return tokens, {}

    while i < len(tokens):
        if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
            novos.append(new_token)
            if i > 0:
                par_antes = (tokens[i - 1], tokens[i])
                contagem_que_importa[par_antes] = (
                    contagem_que_importa.get(par_antes, 0) + 1
                )
            par_proximo = (tokens[i], tokens[i + 1])
            contagem_que_importa[par_proximo] = (
                contagem_que_importa.get(par_proximo, 0) + 1
            )
            i += 2
        else:
            novos.append(tokens[i])
            i += 1

    # contagem_novos = _count_pairs(novos)
    # contagem_que_importa = {}
    # for par, v in contagem_novos.items():
    #     if new_token in par:
    #         contagem_que_importa.update({par: v})
    return (
        novos,
        contagem_que_importa,
    )  # retorna quantos encontrou tambem para contagem


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

    def _pre_tokenize(self, text: str) -> List[int]:
        """
        Faz a pre-tokenizacao e depois ja converte para bytes
        """
        splits = re.findall(re_pretokenize, text)
        b_splits = [list(t.encode("utf8")) for t in splits]
        return b_splits

    def _get_counts(self, tokens: List[List[int]]):
        contagens = {}
        print("CONTANDO OS PARES")
        # aqui so levantamento das estruturas de dados
        contagens_workers = run_pool_async(
            self.pool,
            function=_count_pairs,
            payload=tokens,
            chunk_size=max(len(tokens) // 32, 1_000),
        )
        for c in contagens_workers:
            contagens.update(c)
        priorizacao = [(-i, par) for par, i in contagens.items()]
        heapq.heapify(priorizacao)

        return priorizacao

    def _apply_merge(self, pair, new_token, tokens):
        payload = [[pair, new_token, t] for t in tokens]
        if len(payload) < 1000:
            merges = _merge_single(payload)
        else:
            merges = run_pool_async(
                self.pool, _merge, payload, chunk_size=max(len(payload) // 32, 1_000)
            )

        tokens_merged = []
        contagens_update = {}
        tokens_merged = run_pool_async(
            self.pool, _filtra_resultados, merges, max(len(payload) // 32, 1_000)
        )
        tokens_merged = list(filter(lambda x: len(x) > 1, tokens_merged))
        for result in merges:
            # if len(result[0]) > 1:
            #     tokens_merged.append(result[0])
            nova_contagem = result[1]
            for par, quantidade in nova_contagem.items():
                contagens_update[par] = contagens_update.get(par, 0) + quantidade

        return tokens_merged, contagens_update

    def train_from_texts(self, texts: List[str]):
        tokens = []
        print("PRE_PROCESSANDO")
        for t in tqdm(texts):
            tokens += self._pre_tokenize(t)

        # pega as contagens
        priorizacao = self._get_counts(tokens)

        # agora vamos para os merges
        print("FAZENDO OS MERGES")
        merges = []
        tokens_vocab = [i for i in range(256)]
        for i in tqdm(range(self.vocab_size - 256)):
            if len(tokens) > 0:
                # par_mais = max(contagens, key=contagens.get)
                par_mais = heapq.heappop(priorizacao)[1]
                novo_token = len(tokens_vocab)
                # print(par_mais, "-", novo_token)
                merges.append(par_mais)
                tokens_vocab.append(novo_token)

                tokens, novas_contagens = self._apply_merge(
                    par_mais, novo_token, tokens
                )
                print(f"Novas_contagens: {len(novas_contagens)}\ttokens: {len(tokens)}")

                for par, c in novas_contagens.items():
                    heapq.heappush(priorizacao, (-c, par))

        self.vocab = tokens_vocab
        self.vocab_size = len(self.vocab)
        self.merges = merges
        self.save()

    def save(self):
        obj = {
            "vocab": {self.decode([idx]): idx for idx in self.vocab},
            "merges": [list(m) for m in self.merges],
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
        meus_merges = []
        merges = dados["merges"]
        for idx, par in enumerate(merges):
            m = tuple(par)
            meus_merges.append(m)
            tokens_vocab.append(256 + idx)

        self.vocab = tokens_vocab
        self.merges = meus_merges
        self.vocab_size = len(self.vocab)

    def encode(self, text):
        encoded = []
        tokens = self._pre_tokenize(text)
        print(tokens)
        for conjunto in tokens:
            if len(conjunto) > 1:
                for idx, m in enumerate(self.merges):
                    merged_tokens, _ = self._apply_merge(
                        pair=m, new_token=idx + 256, tokens=[conjunto]
                    )
                    if len(merged_tokens) == 0:
                        encoded.append(256 + idx)
                        break
                    else:
                        conjunto = merged_tokens[0]
                if len(merged_tokens) > 0:
                    encoded += merged_tokens[0]
            else:
                encoded += conjunto
        return encoded

    def decode(self, ids: List[int]):
        if self.bytes_vocab is None:
            bytes_vocab = {idx: bytes([idx]) for idx in range(256)}
            for idx, (p0, p1) in enumerate(self.merges):
                bytes_vocab[idx + 256] = bytes_vocab[p0] + bytes_vocab[p1]

            self.bytes_vocab = bytes_vocab

        tokens = b"".join(self.bytes_vocab[idx] for idx in ids)
        text = tokens.decode("utf8", errors="replace")
        return text


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
    tokenizer = CharyluTokenizer(vocab_size=10000, tokenizer_path="artifacts/charylu")
    tokenizer.train_from_texts(textos)
    tokens = tokenizer.encode(
        "ho ho ho o o o o feliz natal do caralhão Amarelado da porra"
    )
    tokens = [(t, tokenizer.decode([t])) for t in tokens]
    print(tokens)
    # # texto = tokenizer.decode(tokens)
    # # print(texto)
    # print(tokenizer.vocab_size)
