import sys
from Bio import SeqIO



for seq_record in SeqIO.parse(sys.argv[1], "fasta"):
        sequence=str(seq_record.seq)
        if sequence.isupper()==False:    print(sequence)
