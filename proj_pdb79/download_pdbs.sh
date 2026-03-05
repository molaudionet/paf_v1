
mkdir -p data/pdbs
awk -F, 'NR>1{print substr($1,1,4)}' data/kinase_pdbs/kinase_list.csv | sort -u > data/pdb_ids.txt

while read id; do
  curl -L "https://files.rcsb.org/download/${id}.pdb" -o "data/pdbs/${id}.pdb"
done < data/pdb_ids.txt
