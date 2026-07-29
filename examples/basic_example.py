from gris import compute_DepScore_emb

# Example translations
references = ["Der Hund rannte schnell."]
hypotheses = ["Der Hund lief schnell."]

score = compute_DepScore_emb(
    hyps=hypotheses,
    refs=references,
    lang="de"
)

print("GRIS-DepScore:", score)
