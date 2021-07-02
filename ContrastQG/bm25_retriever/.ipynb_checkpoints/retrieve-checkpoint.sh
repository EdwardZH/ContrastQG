export dataset_name=nfcorpus ## scifact ; webis-touche2020 ; hotpotqa ; nfcorpus
export generator_folder=qg_t5-base ## qg_t5-small ; qg_t5-base
export data_path=/home/sunsi/experiments/ContrastQG/$dataset_name/$generator_folder

./bin/SearchCollection -index $dataset_name -topicreader TsvString -topics $data_path/qid2query.tsv -bm25 -output $data_path/bm25_retrieval.trec