from pathlib import Path
import re
from typing import List
import json
import heapq

from tqdm import tqdm

re_pretokenize = r"\d|[^\w\s]{3}|[^\w\s]{1}| ?[^\d\s\W]+|\s{16}|\s{12}|\s{8}|\s{4}|\s"


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

        if tokenizer_path is not None:
            if Path(self.tokenizer_path).exists():
                self.load()
                self.loaded = True

    def _preprocess(self, text: str) -> List[int]:
        """
        Faz a pre-tokenizacao e depois ja converte para bytes
        """
        splits = re.findall(re_pretokenize, text)
        b_splits = [list(t.encode("utf8")) for t in splits]
        return b_splits

    def _get_stats(self, tokens):
        links = {}
        contagens = {}
        indice_global = {}
        global_inverso = {}
        # aqui so levantamento das estruturas de dados
        indice = 0
        for tokens_doc in tokens:
            par_anterior = None
            for idx_par_doc, par in enumerate(zip(tokens_doc, tokens_doc[1:])):
                links[indice] = (par_anterior, None)
                contagens[par] = contagens.get(par, 0) + 1
                indice_global[indice] = par
                # coloca no inverso tambem para ficar facil de econtrar depois
                if par not in global_inverso:
                    global_inverso[par] = set([indice])
                else:
                    global_inverso[par].add(indice)
                # coloca o proximo no cara que ja rodamos
                if idx_par_doc > 0:
                    links[indice - 1] = (links[indice - 1][0], indice)

                par_anterior = indice
                indice += 1

        priorizacao = [(-i, par) for par, i in contagens.items()]
        heapq.heapify(priorizacao)
        # return links, contagens, indice_global, global_inverso
        return links, priorizacao, indice_global, global_inverso

    def _merge(
        self, par_mais, novo_token, links, contagens, indice_global, global_inverso
    ):
        """
        Precisa atualizar os links tambem. Quando tira alguem do meio precisa linkar os caras que estao em volta
        """
        # durante o encoding pode ser que pegamos um trecho que nao existe
        if par_mais in global_inverso:
            para_remover_global_inverso = []
            alterados = (
                set()
            )  # quando temos caras repetidos um seguido do outro, vamos alterar caras que estao no para mudar
            # se isso acontecer, vamos so pular os caras alterados
            # primeiro acha todo mundo que tem que mudar
            para_mudar = global_inverso[par_mais]
            for indice_muda in para_mudar:
                if indice_muda in alterados:
                    continue
                # para cada cara desse vamos ver se tem gente antes que precisa mudar
                link_muda = links[indice_muda]
                # vamos ja compatibilizar os indices,
                if (
                    link_muda["anterior"] is not None
                    and link_muda["proximo"] is not None
                ):
                    # faz uma ligacao direta
                    indice_anterior = link_muda["anterior"]
                    indice_proximo = link_muda["proximo"]
                    links[indice_anterior]["proximo"] = indice_proximo
                    links[indice_proximo]["anterior"] = indice_anterior
                elif link_muda["anterior"] is not None:
                    # so apaga o proximo desse cara
                    links[link_muda["anterior"]]["proximo"] = None
                elif link_muda["proximo"] is not None:
                    # so apaga o anterior desse cara
                    links[link_muda["proximo"]]["anterior"] = None
                del links[indice_muda]

                if link_muda["anterior"] is not None:
                    # vai ate ele e muda o token da direita
                    par_que_muda = indice_global[link_muda["anterior"]]
                    # antes de mudar ja atualiza a contagem
                    contagens[par_que_muda] -= 1
                    if contagens[par_que_muda] == 0:
                        del contagens[par_que_muda]

                    # atualiza o indice inverso tambem
                    # global_inverso[par_que_muda].remove(link_muda["anterior"])
                    # guarda para remover depois
                    para_remover_global_inverso.append(
                        (par_que_muda, link_muda["anterior"])
                    )
                    # agora muda o cara
                    par_que_muda = (par_que_muda[0], novo_token)
                    # devolve ele no indice global
                    indice_global[link_muda["anterior"]] = par_que_muda
                    # atualiza indice inverso com o cara novo
                    if par_que_muda not in global_inverso:
                        global_inverso[par_que_muda] = set([link_muda["anterior"]])
                    else:
                        global_inverso[par_que_muda].add(link_muda["anterior"])
                    # contagens
                    contagens[par_que_muda] = contagens.get(par_que_muda, 0) + 1
                    # nao esquece de registrar que mudou para o caso dos repetidos seguidos
                    alterados.add(link_muda["anterior"])

                # mesma coisa mas mudando o que vem depois
                if link_muda["proximo"] is not None:
                    # vai ate ele e muda o token da esquerda
                    par_que_muda = indice_global[link_muda["proximo"]]
                    # antes de mudar ja atualiza a contagem
                    contagens[par_que_muda] -= 1
                    if contagens[par_que_muda] == 0:
                        del contagens[par_que_muda]

                    # atualiza o indice inverso tambem
                    # global_inverso[par_que_muda].remove(link_muda["proximo"])
                    # guarda para remover depois
                    para_remover_global_inverso.append(
                        (par_que_muda, link_muda["proximo"])
                    )
                    # agora muda o cara
                    par_que_muda = (novo_token, par_que_muda[1])
                    # devolve ele no indice global
                    indice_global[link_muda["proximo"]] = par_que_muda
                    # atualiza indice inverso com o cara novo
                    if par_que_muda not in global_inverso:
                        global_inverso[par_que_muda] = set([link_muda["proximo"]])
                    else:
                        global_inverso[par_que_muda].add(link_muda["proximo"])
                    # contagens
                    contagens[par_que_muda] = contagens.get(par_que_muda, 0) + 1
                    # nao esquece de registrar que mudou para o caso dos repetidos seguidos
                    alterados.add(link_muda["proximo"])

                # por ultimo precisa apagar o cara que foi mergeado de todas as estatisticas
                # do inverso tem que ser aqui dentro do loop
                # global_inverso[par_mais].remove(indice_muda)
                # guarda para remover depois
                para_remover_global_inverso.append((par_mais, indice_muda))
                del indice_global[indice_muda]
            # agora sim apaga do indice inverso
            for par, indice in para_remover_global_inverso:
                # if indice in global_inverso[par]:
                global_inverso[par].remove(indice)
            # global_inverso = self._create_inverse_global(indice_global)

            del contagens[par_mais]
        return links, contagens, indice_global, global_inverso

    def _merge_(self, par_mais, novo_token, links, indice_global, global_inverso):
        """
        Precisa atualizar os links tambem. Quando tira alguem do meio precisa linkar os caras que estao em volta
        """
        # durante o encoding pode ser que pegamos um trecho que nao existe
        novas_contagens = {}
        if par_mais in global_inverso:
            para_remover_global_inverso = []
            alterados = (
                set()
            )  # quando temos caras repetidos um seguido do outro, vamos alterar caras que estao no para mudar
            # se isso acontecer, vamos so pular os caras alterados
            # primeiro acha todo mundo que tem que mudar
            para_mudar = global_inverso[par_mais]
            for indice_muda in para_mudar:
                if indice_muda in alterados:
                    continue
                # para cada cara desse vamos ver se tem gente antes que precisa mudar
                link_muda = links[indice_muda]
                indice_anterior = link_muda[0]
                indice_proximo = link_muda[1]
                # vamos ja compatibilizar os indices,
                if indice_anterior is not None and indice_proximo is not None:
                    # faz uma ligacao direta
                    links[indice_anterior] = (
                        links[indice_anterior][0],
                        indice_proximo,
                    )
                    links[indice_proximo] = (
                        indice_anterior,
                        links[indice_proximo][1],
                    )
                elif indice_anterior is not None:
                    # so apaga o proximo desse cara
                    links[indice_anterior] = (links[indice_anterior][0], None)
                elif indice_proximo is not None:
                    # so apaga o anterior desse cara
                    links[indice_proximo] = (None, links[indice_proximo][1])
                del links[indice_muda]

                if indice_anterior is not None:
                    # vai ate ele e muda o token da direita
                    par_que_muda = indice_global[indice_anterior]

                    # atualiza o indice inverso tambem
                    # global_inverso[par_que_muda].remove(indice_anterior)
                    # guarda para remover depois
                    para_remover_global_inverso.append((par_que_muda, indice_anterior))
                    # agora muda o cara
                    par_que_muda = (par_que_muda[0], novo_token)
                    # devolve ele no indice global
                    indice_global[indice_anterior] = par_que_muda
                    # atualiza indice inverso com o cara novo
                    if par_que_muda not in global_inverso:
                        global_inverso[par_que_muda] = set([indice_anterior])
                    else:
                        global_inverso[par_que_muda].add(indice_anterior)
                    # contagens
                    novas_contagens[par_que_muda] = (
                        novas_contagens.get(par_que_muda, 0) + 1
                    )
                    # nao esquece de registrar que mudou para o caso dos repetidos seguidos
                    alterados.add(indice_anterior)

                # mesma coisa mas mudando o que vem depois
                if indice_proximo is not None:
                    # vai ate ele e muda o token da esquerda
                    par_que_muda = indice_global[indice_proximo]

                    # atualiza o indice inverso tambem
                    # global_inverso[par_que_muda].remove(indice_proximo)
                    # guarda para remover depois
                    para_remover_global_inverso.append((par_que_muda, indice_proximo))
                    # agora muda o cara
                    par_que_muda = (novo_token, par_que_muda[1])
                    # devolve ele no indice global
                    indice_global[indice_proximo] = par_que_muda
                    # atualiza indice inverso com o cara novo
                    if par_que_muda not in global_inverso:
                        global_inverso[par_que_muda] = set([indice_proximo])
                    else:
                        global_inverso[par_que_muda].add(indice_proximo)
                    # contagens
                    novas_contagens[par_que_muda] = (
                        novas_contagens.get(par_que_muda, 0) + 1
                    )
                    # nao esquece de registrar que mudou para o caso dos repetidos seguidos
                    alterados.add(indice_proximo)

                # por ultimo precisa apagar o cara que foi mergeado de todas as estatisticas
                # do inverso tem que ser aqui dentro do loop
                # global_inverso[par_mais].remove(indice_muda)
                # guarda para remover depois
                para_remover_global_inverso.append((par_mais, indice_muda))
                del indice_global[indice_muda]
            # agora sim apaga do indice inverso
            for par, indice in para_remover_global_inverso:
                # if indice in global_inverso[par]:
                global_inverso[par].remove(indice)
                if len(global_inverso[par]) == 0:
                    del global_inverso[par]

        return links, novas_contagens, indice_global, global_inverso

    def _create_inverse_global(self, indice_global):
        inverse = {}
        for idx, pair in indice_global.items():
            if pair in inverse:
                inverse[pair].add(idx)
            else:
                inverse[pair] = set([idx])
        return inverse

    def train_from_texts(self, texts: List[str]):
        tokens = []
        print("PRE_PROCESSANDO")
        for t in tqdm(texts):
            tokens += self._preprocess(t)

        print("CALCULANDO ESTATISTICAS")
        # links, contagens, indice_global, global_inverso = self._get_stats(tokens)
        links, priorizacao, indice_global, global_inverso = self._get_stats(tokens)

        tokens = None
        del tokens  # libera memoria

        # agora vamos para os merges
        print("FAZENDO OS MERGES")
        merges = []
        tokens_vocab = [i for i in range(256)]
        for i in tqdm(range(self.vocab_size - 256)):
            if len(priorizacao) > 0:
                # par_mais = max(contagens, key=contagens.get)
                par_mais = heapq.heappop(priorizacao)[1]
                novo_token = len(tokens_vocab)
                # print(par_mais, "-", novo_token)
                merges.append(par_mais)
                tokens_vocab.append(novo_token)

                # agora atualiza todos os dados
                # links, contagens, indice_global, global_inverso = self._merge(
                #     par_mais,
                #     novo_token,
                #     links,
                #     contagens,
                #     indice_global,
                #     global_inverso,
                # )

                links, novas_contagens, indice_global, global_inverso = self._merge_(
                    par_mais,
                    novo_token,
                    links,
                    indice_global,
                    global_inverso,
                )
                print(
                    f"links: {len(links)}\novas_contagens: {len(novas_contagens)}\tindice_global: {len(indice_global)}\tglobal_inverso: {len(global_inverso)}"
                )

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
        tokens = self._preprocess(text)
        print(tokens)
        for conjunto in tokens:
            if len(conjunto) > 1:
                links, priorizacao, indice_global, global_inverso = self._get_stats(
                    [conjunto]
                )

                for idx, m in enumerate(self.merges):
                    # so faz o merge se existir o par para mergear
                    if m in global_inverso:
                        links, novas_contagens, indice_global, global_inverso = (
                            self._merge_(
                                m,
                                256 + idx,
                                links,
                                indice_global,
                                global_inverso,
                            )
                        )
                        # pode ser que o cara sumarizou o conjunto todo e fique tudo vazio
                        # virou o novo token apenas, nem tem par
                        if len(indice_global) == 0:
                            indice_global[0] = (256 + idx,)
                            break

                for idx, (_, par) in enumerate(indice_global.items()):
                    if idx == 0:
                        encoded += list(par)
                    elif len(par) > 1:
                        encoded.append(par[1])
                    else:
                        encoded.append(par[0])
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
        "uma Iniciativa da literatura bemmelhante! huhu",
        "uma nova historia emocionante da literatura mundial",
        "o rato roeu a roupa do rei de roma aquele arrombadinho do caralho",
        "eu gósto de falar com açentos malúcos enquânto isso!",
        """
0

Login
Login
CPF
Guia Enem
 
Língua Portuguesa
Matemática
Geografia
História
Física
Química
Biologia
Inglês
Educação Física
Artes
Religião
Educação Financeira
Espanhol
Disciplinas Complementares
Antropologia
Astronomia
Datas Comemorativas
Filosofia
Sociologia
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
WhatsAppSeg a Sex: 07h às 20h40
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
Graduação / Pós
Educação Básica
Cursos Técnicos
Idioma / Profissionalizantes
Treinamento
Terceirizado
Quero ser Parceiro
Prêmio Reclame Aqui
© 2024  Educa Mais Brasil. Todos os direitos reservados
Utilizamos cookies.  Política de Privacidade.
Quer saber de um jeito para fazer faculdade com desconto sem precisar do Enem? 😌""",
    ]
    tokenizer = CharyluTokenizer(vocab_size=10000, tokenizer_path="artifacts/charylu")
    tokenizer.train_from_texts(textos)
    tokens = tokenizer.encode(
        "ho ho ho o o o o feliz natal do caralhão Amarelado da porra"
    )
    tokens = [(t, tokenizer.decode([t])) for t in tokens]
    print(tokens)
    # texto = tokenizer.decode(tokens)
    # print(texto)
    print(tokenizer.vocab_size)
