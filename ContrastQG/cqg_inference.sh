# !/bin/bash
## --------------------------------------------
export CUDA_VISIBLE_DEVICES=$2
pretrain_generator_type=t5-base ## t5-small ; t5-base
batch_size=128 ## 200; 400
 ## scifact ; webis-touche2020 ; hotpotqa ; nfcorpus
generator_mode=qg
cgenerator_mode=cqg
model_dir=../checkpoints/$generator_mode-$pretrain_generator_type/checkpoint
cmodel_dir=../checkpoints/$cgenerator_mode-$pretrain_generator_type/checkpoint
input_dir=../datasets
output_dir=../

task=$1
if [ ! -d $output_dir/bm25_index/$task ]; then
  mkdir -p -m 755 $output_dir/bm25_index/$task
fi

python sample_docs.py --input_path $input_dir --task $task --output_path $output_dir
python inference.py \
--generator_mode $generator_mode \
--pretrain_generator_type $pretrain_generator_type \
--batch_size $batch_size \
--model_dir $model_dir \
--task $task \
--input_dir $input_dir \
--output_dir $output_dir \
--save_txt



./bm25_retriever/bin/IndexCollection -collection JsonCollection -input $output_dir/$generator_mode\_$pretrain_generator_type/$task/corpus -index $output_dir/bm25_index/$task -generator LuceneDocumentGenerator -threads 8 -storePositions -storeDocvectors -storeRawDocs
./bm25_retriever/bin/SearchCollection -index $output_dir/bm25_index/$task -topicreader TsvString -topics $output_dir/$generator_mode\_$pretrain_generator_type/$task/queries.txt -bm25 -output $output_dir/$generator_mode\_$pretrain_generator_type/$task/bm25.trec -hits 100
python  sample_contrast_pairs.py --input_path $output_dir/$generator_mode\_$pretrain_generator_type/$task --output_path $output_dir/$cgenerator_mode\_$pretrain_generator_type/$task

python inference.py \
--generator_mode $cgenerator_mode \
--pretrain_generator_type $pretrain_generator_type \
--batch_size $batch_size \
--model_dir $cmodel_dir \
--task $task \
--input_dir $input_dir \
--output_dir $output_dir \
--num_return_sequences 1 \
--do_sample
