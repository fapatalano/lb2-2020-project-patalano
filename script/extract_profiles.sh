#!/bin/bash

input_path=$1
output_path=$2
for file in $(ls $input_path);do
  id=${file:0:-5}
  python3 extract_profile.py $input_path/$file > $output_path/$id.prf

echo "$id done "
done
