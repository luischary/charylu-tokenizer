from pathlib import Path
import os

from charylutokenizer.charylutokenizer import CharyluTokenizer


def load(vocab_size: int, version: str = "") -> CharyluTokenizer:
    """
    Loads up a tokenizer version.

    - Vocab size must be one of [32, 50, 60, 70, 80, 90, 100, 110, 120, 130, 150].
    The real vocab size is this value * 1000.
    - Version: wich version to use ["", "_nocode"]. One version was trained
    with code texts and the other was not.
    """
    artifact_path = f"{os.path.dirname(__file__)}/artifacts/charylu{version}/tokenizer_2024_{vocab_size}k.json"
    if not Path(artifact_path).exists():
        raise NotImplementedError("The combination of parameters is incorrect!")

    tokenizer = CharyluTokenizer(artifact_path)
    return tokenizer
