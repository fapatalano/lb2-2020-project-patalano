#!/bin/bash

profile_id=$1
input_path=$2
output_path=$3


 while read line; do
    mv $input_path/$line $output_path

echo "$line done"
done < $1
