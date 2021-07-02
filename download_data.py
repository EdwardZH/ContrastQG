from beir import util
from beir.datasets.data_loader import GenericDataLoader

for dataset in ["climate-fever"]:
    print (dataset)
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
    data_path = util.download_and_unzip(url, "datasets")
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")