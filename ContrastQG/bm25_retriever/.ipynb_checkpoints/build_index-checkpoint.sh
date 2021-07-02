export dataset_name=nfcorpus ## scifact ; webis-touche2020 ; hotpotqa ; nfcorpus
export data_path=/home/sunsi/experiments/ContrastQG/$dataset_name

./bin/IndexCollection -collection JsonCollection -input $data_path/corpus -index $dataset_name -generator LuceneDocumentGenerator -threads 8 -storePositions -storeDocvectors -storeRawDocs