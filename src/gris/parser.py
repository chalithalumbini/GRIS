"""
Fixed dependency parser using Stanza.
Handles multi-sentence input (maintains token id offsets within each input text).
"""

from dataclasses import dataclass
from typing import List, Optional, Dict
import stanza


@dataclass
class DepToken:
    id: int
    text: str
    lemma: str
    upos: str
    head: int
    deprel: str
    feats: Optional[Dict[str, str]] = None


@dataclass
class DepSentence:
    text: str
    tokens: List[DepToken]

    def __len__(self):
        return len(self.tokens)


class StanzaDependencyParser:
    """
    Wrapper for Stanza dependency parser.
    Maintains proper token ID offsets across sentence boundaries within a single input string.
    """

    # Process-local cache to avoid redundant stanza.download() checks and
    # expensive Pipeline construction when callers instantiate repeatedly.
    _PIPELINES: Dict[tuple, stanza.Pipeline] = {}
    _DOWNLOADED: set[str] = set()

    def __init__(self, lang: str = "en", use_gpu: bool = False):
        key = (lang, bool(use_gpu))

        # Download resources once per language per process.
        if lang not in self.__class__._DOWNLOADED:
            stanza.download(lang, processors="tokenize,pos,lemma,depparse", verbose=False)
            self.__class__._DOWNLOADED.add(lang)

        # Reuse Pipeline if already constructed.
        if key not in self.__class__._PIPELINES:
            self.__class__._PIPELINES[key] = stanza.Pipeline(
                lang=lang,
                processors="tokenize,pos,lemma,depparse",
                use_gpu=use_gpu,
                logging_level="ERROR",
            )

        self.nlp = self.__class__._PIPELINES[key]

    def parse(self, sentences: List[str]) -> List[DepSentence]:
        output: List[DepSentence] = []

        for text in sentences:
            doc = self.nlp(text)
            tokens: List[DepToken] = []
            offset = 0

            for sent in doc.sentences:
                for w in sent.words:
                    feats: Dict[str, str] = {}
                    if w.feats:
                        for pair in w.feats.split("|"):
                            if "=" in pair:
                                k, v = pair.split("=", 1)
                                feats[k] = v

                    tokens.append(
                        DepToken(
                            id=w.id + offset,
                            text=w.text,
                            lemma=w.lemma or w.text,
                            upos=w.upos or "X",
                            head=(w.head + offset) if w.head else 0,
                            deprel=w.deprel or "dep",
                            feats=feats or None,
                        )
                    )

                offset += len(sent.words)

            output.append(DepSentence(text=text, tokens=tokens))

        return output