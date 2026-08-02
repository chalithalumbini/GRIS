from gris import compute_DepScore_emb

# Example translations
references = ["Sie kÃ¶nnen jederzeit wiederkommen, da unser Chat-Service-Fenster tÃ¤glich rund um die Uhr geÃ¶ffnet ist"]
hypotheses = ["Sie kÃ¶nnen jederzeit zurÃ¼ckkehren, da unser Chat-Service-Fenster rund um die Uhr geÃ¶ffnet ist"]

score = compute_DepScore_emb(
    hyps=hypotheses,
    refs=references,
    lang="de",
    explain=True,
)

print("GRIS-DepScore:", score)