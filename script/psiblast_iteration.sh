#!/bin/bash

for file in $(ls ../training_ds/fasta/);do
# for file in $(ls ../blind_test_set/fasta_150/);do
        string=$file
        id=${string:0:-6}
        # psiblast -query ../blind_test_set/fasta_150/$file -db ../swissprot/uniprot_sprot.fasta -evalue 0.01 -num_iterations 3 -out_ascii_pssm ../psiblast_output/pssm/pssm_blind_test_set/$id.pssm \
        # -num_descriptions 10000 -num_alignments 10000 -out ../psiblast_output/psiblast_blind_test_set/$id.alns.blast
        psiblast -query ../training_ds/fasta/$file -db ../swissprot/uniprot_sprot.fasta -evalue 0.01 -num_iterations 3 -out_ascii_pssm ../psiblast_output/pssm/$id.pssm \
        -num_descriptions 10000 -num_alignments 10000 -out ../psiblast_output/$id.alns.blast


echo "$string done "
done
