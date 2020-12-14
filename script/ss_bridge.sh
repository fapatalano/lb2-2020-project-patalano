for file in $(ls ../fasta_150);do
  echo "$file"
  python3 lowercase.py ../fasta_150/$file
#echo "$file done "
done
