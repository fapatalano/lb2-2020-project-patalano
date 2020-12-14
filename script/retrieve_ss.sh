#!/bin/bash

for file in $(ls ../dssp_150);do
        string=$file
        id=${string:0:-5}
        id=${id^^}
        # python3 dssp.py ../dssp_150/$string
        python3 dssp.py ../dssp_150/$string  > ..//blind_test_set/dssp_150/$id.dssp
        # python3 dssp.py ../dssp_150/$string > ../fasta_150/$id.fasta

echo "$string done "
done
