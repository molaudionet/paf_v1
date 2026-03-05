# Ligand–Pocket Retrieval Summary (v1)

- n pairs: 50
- pocket params: radius=10.0, gamma_fm=0.15, sigma_t=0.04
- ligand embed: K=256, gamma_fm=0.15, sigma_t=0.04
- cross-spectrum scoring: OFF

## Easy negatives
```json
{
  "n": 50,
  "top1_cosine": 0.02,
  "top5_cosine": 0.08,
  "top10_cosine": 0.16,
  "mrr_cosine": 0.08090238781159456,
  "mean_auroc_cosine": 0.4861224489795919
}
```

## Hard negatives
```json
{
  "n": 50,
  "top1_cosine": 0.02,
  "top5_cosine": 0.08,
  "top10_cosine": 0.16,
  "mrr_cosine": 0.08103262036973408,
  "mean_auroc_cosine": 0.4861224489795919
}
```

## Files
- retrieval_table.csv
- metrics_easy.json
- metrics_hard.json
- auroc_curves.png
- mrr_by_family.png
