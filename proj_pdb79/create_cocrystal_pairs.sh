python - <<'PY'
import pandas as pd

src = "./data/kinase_list.csv"
df = pd.read_csv(src)

out = pd.DataFrame({
    "pdb_path": df["pdb_path"],
    "protein_chain": df["chain_id"],
    "ligand_resname": df["ligand_resname"],
    "family": df["family"]
})

out.to_csv("data/cocrystal_pairs.csv", index=False)

print("Wrote data/cocrystal_pairs.csv")
print(out.head())
PY
