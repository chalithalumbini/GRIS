import spacy
import benepar


class ConstituencyParser:
    def __init__(self, lang="en"):
        self.enabled = False
        self.nlp = None

        print(f"[INFO] Loading Benepar constituency parser for '{lang}'...")

        if lang == "en":
            try:
                self.nlp = spacy.load("en_core_web_sm")
                if "benepar" not in self.nlp.pipe_names:
                    self.nlp.add_pipe("benepar", config={"model": "benepar_en3"})
                self.enabled = True
            except Exception as e:
                print(f"[WARN] Failed to load Benepar; constituency parsing disabled: {e}")
        else:
            print(f"[INFO] Constituency parsing not supported for lang='{lang}'. Skipping.")

    def parse(self, sents):
        # Always return same shape: List[List[str or None]]
        if not self.enabled:
            return [[None for _ in s.split(".") if _.strip()] for s in sents]

        docs = list(self.nlp.pipe(sents))
        trees = []

        for doc in docs:
            sent_trees = []
            for sent in doc.sents:
                try:
                    tree = sent._.parse_string
                except Exception as e:
                    print(f"[WARN] Failed to get constituency parse: {e}")
                    tree = None
                sent_trees.append(tree)
            trees.append(sent_trees)

        return trees
