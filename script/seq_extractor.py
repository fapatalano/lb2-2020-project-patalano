from Bio import SeqIO
import sys

wanted = [line.strip() for line in open(sys.argv[2])]
seqiter = SeqIO.parse(open(sys.argv[1]), 'fasta')
for seq in seqiter:
    if seq.id[:6] in wanted:
        print('>'+seq.id[:6])
        print(seq.seq)
