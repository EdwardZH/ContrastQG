export CUDA_VISIBLE_DEVICES=$2
pretrain_generator_type=t5-base ## t5-small ; t5-base
batch_size=32 ## 200; 400
generator_mode=qg
cgenerator_mode=cqg
model_dir=../checkpoints/$generator_mode-$pretrain_generator_type/checkpoint
cmodel_dir=../checkpoints/$cgenerator_mode-$pretrain_generator_type/checkpoint
input_dir=../datasets
output_dir=../
task=$1


python sample_docs.py --input_path $input_dir --task $task --output_path $output_dir
python inference.py \
--generator_mode $generator_mode \
--pretrain_generator_type $pretrain_generator_type \
--batch_size $batch_size \
--model_dir $model_dir \
--task $task \
--input_dir $input_dir \
--output_dir $output_dir \
--num_return_sequences 1 \
--do_sample
