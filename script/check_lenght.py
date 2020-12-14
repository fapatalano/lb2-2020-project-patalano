import sys
from Bio import SeqIO


for seq_record in SeqIO.parse(sys.argv[1], "fasta"):
        sequence=str(seq_record.seq)
        if len(sequence)<50 or len(sequence)>300 :    print(seq_record.id)
