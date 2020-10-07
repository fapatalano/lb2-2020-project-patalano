#!/bin/bash

filename='id_pdb_150_newlines.txt'

while read line; do
	pdb="$line"
	id=${pdb:0:4}
	id=${id,,}
	chain="${pdb:(-1)}"
	pdb_selchain -$chain ./pdb/$id.pdb > ./chains/$id\_$chain.pdb

echo "finish"
done < $filename
