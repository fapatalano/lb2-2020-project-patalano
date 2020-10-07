#!/bin/bash

input_path=$1
output_path=$2

for file in $(ls $input_path);do
        string="$file"
        id=${string:0:-4}
        mkdssp -i $input_path/$file -o $output_path/$id.dssp

echo "$id done"
done
