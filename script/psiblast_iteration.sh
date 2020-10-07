#!/bin/bash

for file in $(ls ../blind_test_set/fasta_150/);do
        string=$file
        id=${string:0:-6}
        psiblast -query ../blind_test_set/fasta_150/$file -db ../swissprot/uniprot_sprot.fasta -evalue 0.01 -num_iterations 3 -out_ascii_pssm ../psiblast_output/pssm/pssm_blind_test_set/$id.pssm \
        -num_descriptions 10000 -num_alignments 10000 -out ../psiblast_output/psiblast_blind_test_set/$id.alns.blast -comp_based_stats No


echo "$string done "
done
