import json

data_dict = {}
with open("/home2/liuzhenghao/beir_data/cqg_t5-base/nfcorpus/queries.jsonl") as fin:
    for line in fin:
        data = json.loads(line)
        id = str(data["pos_doc_id"]) + " " + str(data["neg_doc_id"])
        data_dict[id] = data["text"]
with open("/home2/liuzhenghao/beir_data/cqg_t5-base/nfcorpus1/queries.jsonl") as fin:
    for line in fin:
        data = json.loads(line)
        id = str(data["pos_doc_id"]) + " " + str(data["neg_doc_id"])
        if id in data_dict:
            print (id)
            print (data["_id"])
            print (data_dict[id])


