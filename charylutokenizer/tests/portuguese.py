import re

from typing import List

from tokenizers import NormalizedString, PreTokenizedString

prefixos = [
    "contra",
    "ambi",
    "anti",
    "ante",
    "semi",
    "bem",
    "ben",
    "bis",
    "pos",
    "tri",
    "in",
    "bi",
    "i",
]

suffixos = [
    "escer",
    "mente",
    "alhão",
    "mento",
    "iscar",
    "ência",
    "douro",
    "tório",
    "tério",
    "ismo",
    "tica",
    "agem",
    "ário",
    "eiro",
    "aria",
    "isco",
    "eiro",
    "ista",
    "dade",
    "eira",
    "inho",
    "acho",
    "icho",
    "ento",
    "aréu",
    "arra",
    "into",
    "enho",
    "onho",
    "iaco",
    "ismo",
    "aico",
    "arrão",
    "eirão",
    "ense",
    "ista",
    "ável",
    "ível",
    "óvel",
    "úvel",
    "agem",
    "ança",
    "eria",
    "ença",
    "ejar",
    "icar",
    "itar",
    "ecer",
    "ano",
    "ena",
    "eno",
    "ino",
    "iço",
    "ivo",
    "ude",
    "dor",
    "iam",
    "ndo",
    "aço",
    "uça",
    "tor",
    "eco",
    "ela",
    "ote",
    "sor",
    "ume",
    "nte",
    "rio",
    "ura",
    "eza",
    "ice",
    "ado",
    "ato",
    "ada",
    "ama",
    "ica",
    "ame",
    "edo",
    "oso",
    "udo",
    "aco",
    "ata",
    "ção",
    "ear",
    "az",
    "ez",
    "am",
    "ão",
    "eo",
    "ão",
    "or",
    "ia",
    "al",
    "ês",
    "eu",
    "s",
]


# re_suffix = r"escer\b|mente\b|alhão\b|mento\b|iscar\b|ência\b|douro\b|tório\b|tério\b|ismo\b|tica\b|agem\b|ário\b|eiro\b|aria\b|isco\b|eiro\b|ista\b|dade\b|eira\b|inho\b|acho\b|icho\b|ento\b|aréu\b|arra\b|into\b|enho\b|onho\b|iaco\b|ismo\b|aico\b|arrão\b|eirão\b|ense\b|ista\b|ável\b|ível\b|óvel\b|úvel\b|agem\b|ança\b|eria\b|ença\b|ejar\b|icar\b|itar\b|ecer\b|ano\b|ena\b|eno\b|ino\b|iço\b|ivo\b|ude\b|dor\b|iam\b|ndo\b|aço\b|uça\b|tor\b|eco\b|ela\b|ote\b|sor\b|ume\b|nte\b|rio\b|ura\b|eza\b|ice\b|ado\b|ato\b|ada\b|ama\b|ica\b|ame\b|edo\b|oso\b|udo\b|aco\b|ata\b|ção\b|ear\b|az\b|ez\b|am\b|ão\b|eo\b|ão\b|or\b|ia\b|al\b|ês\b|eu\b|s\b"

# re_preffix = r" ?\bambi| ?\banti| ?\bante| ?\bbem| ?\bben| ?\bbi| ?\bbis| ?\bcontra| ?\bin| ?\bi| ?\bpos| ?\bsemi| ?\btri"

# re_pretokenize = r"\d|[^\w\s]{3}|[^\w\s]{1}| ?[^\d\s\W]+|\s{16}|\s{12}|\s{8}|\s{4}|\s"


class PortuguesePreTokenizer:
    def __init__(self):
        self.re_suffix = r"escer\b|mente\b|alhão\b|mento\b|iscar\b|ência\b|douro\b|tório\b|tério\b|ismo\b|tica\b|agem\b|ário\b|eiro\b|aria\b|isco\b|eiro\b|ista\b|dade\b|eira\b|inho\b|acho\b|icho\b|ento\b|aréu\b|arra\b|into\b|enho\b|onho\b|iaco\b|ismo\b|aico\b|arrão\b|eirão\b|ense\b|ista\b|ável\b|ível\b|óvel\b|úvel\b|agem\b|ança\b|eria\b|ença\b|ejar\b|icar\b|itar\b|ecer\b|ano\b|ena\b|eno\b|ino\b|iço\b|ivo\b|ude\b|dor\b|iam\b|ndo\b|aço\b|uça\b|tor\b|eco\b|ela\b|ote\b|sor\b|ume\b|nte\b|rio\b|ura\b|eza\b|ice\b|ado\b|ato\b|ada\b|ama\b|ica\b|ame\b|edo\b|oso\b|udo\b|aco\b|ata\b|ção\b|ear\b|az\b|ez\b|am\b|ão\b|eo\b|ão\b|or\b|ia\b|al\b|ês\b|eu\b|s\b"
        self.re_preffix = r" ?\bambi| ?\banti| ?\bante| ?\bbem| ?\bben| ?\bbi| ?\bbis| ?\bcontra| ?\bin| ?\bi| ?\bpos| ?\bsemi| ?\btri"
        self.re_pretokenize = (
            r"\d|[^\w\s]{3}|[^\w\s]{1}| ?[^\d\s\W]+|\s{16}|\s{12}|\s{8}|\s{4}|\s"
        )

    def portuguese_split(
        self, i: int, normalized_string: NormalizedString
    ) -> List[NormalizedString]:
        splits = []
        texto = str(normalized_string)
        splits_pre = re.findall(self.re_preffix, texto, flags=re.IGNORECASE)
        if len(splits_pre) > 0:
            # pega a primeira que encontrar
            achou = splits_pre[0]
            splits.append(achou)
            tamanho_achou = len(achou)
            # remove do texto
            texto = texto[tamanho_achou:]
        # agora o suffixo
        splits_pos = re.findall(self.re_suffix, texto)
        if len(splits_pos) > 0:
            achou = splits_pos[0]
            tamanho_achou = len(achou)
            texto = texto[:-tamanho_achou]
            if len(texto) > 0:
                splits.append(texto)
            splits.append(achou)
        else:
            splits.append(texto)

        splits = [NormalizedString(s) for s in splits]
        return splits

    def generic_split(
        self, i: int, normalized_string: NormalizedString
    ) -> List[NormalizedString]:
        splits = re.findall(self.re_pretokenize, str(normalized_string))
        splits = [NormalizedString(s) for s in splits]
        return splits

    def pre_tokenize(self, pretok: PreTokenizedString):
        # primeiro o mais geral
        pretok.split(self.generic_split)

        # agora o focado no portugues
        pretok.split(self.portuguese_split)
