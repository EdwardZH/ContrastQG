# !/bin/bash
## --------------------------------------------
input_dir=/home2/liuzhenghao/beir_data/datasets
output_dir=/home2/liuzhenghao/beir_data

task=$1
if [ ! -d $output_dir/bm25_index/$task ]; then
  mkdir -p -m 755 $output_dir/bm25_results/$task
fi

python  generate_raw_queries.py --input_path $input_dir/$task --output_path $output_dir/bm25_results/$task
./bm25_retriever/bin/SearchCollection -index $output_dir/bm25_index/$task -topicreader TsvString -topics $output_dir/bm25_results/$task/queries.txt -bm25 -output $output_dir/bm25_results/$task/bm25.trec -hits 100
