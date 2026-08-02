from gris import compute_DepScore_emb

# Example translations
references = ["The doctor recovered the patient"]
hypotheses = ["The physicial recovered the patient "]

score = compute_DepScore_emb(
    hyps=hypotheses,
    refs=references,
    lang="en",
    explain=True,
)

print("GRIS-DepScore:", score)