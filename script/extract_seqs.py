from Bio import SeqIO
from random import sample
import sys


#################### this script needs to randomly extract some sequences 
with open(sys.argv[1]) as f:
    # seqs = SeqIO.parse(f,"fasta")
    # print(sample(list(seqs), 2))
    seqs = SeqIO.parse(f, "fasta")
    samps = ((seq.name, seq.seq) for seq in  sample(list(seqs),2))
    for samp in samps:
        print(">{}\n{}".format(*samp))
