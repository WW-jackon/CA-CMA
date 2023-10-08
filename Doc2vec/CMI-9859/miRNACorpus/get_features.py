import numpy as np
import warnings
warnings.filterwarnings('ignore')  
from scipy.ndimage import measurements
import gensim
from multiprocessing import Pool, cpu_count
from Bio import SeqIO






lncrna_model_path = "miRNA_doc2vec_test.model"
lncrna_doc2vec_model = gensim.models.Doc2Vec.load(lncrna_model_path)

def segment(seq):
    res = []
    i = 0
    while i + 3 < len(seq):
        tmp = seq[i:i + 3]
        res.append(tmp)
        i = i + 1
    return res


def lncrna_doc2vec_embedding(seq):
    seg = segment(seq)
    lncrna_doc2vec_model.random.seed(0)
    vec = lncrna_doc2vec_model.infer_vector(seg)
    
    return vec









def read_fa(path):
    res = {}
    rescords = list(SeqIO.parse(path, format="fasta"))
    for x in rescords:
        id = str(x.id)
        
        seq = str(x.seq)
        res[id] = seq
    return res


def to_dict(seq_dict, feature_list):
    res_dict = {}
    for i, k in enumerate(list(seq_dict.keys())):
        res_dict[k] = feature_list[i]
    return res_dict


def save_dict(x_dict, path):
    f = open(path, "w")
    for k, v in x_dict.items():
        tmp = k + "," + ",".join([str(x) for x in v])
        f.write(tmp + "\n")
    f.close()


def load_dict(path):
    lines = open(path, "r").readlines()
    res = {}
    for line in lines:
        x_list = line.strip().split(",")
        id = str(x_list[0])
        vec = [np.float(x) for x in x_list[1:]]
        res[id] = vec
    return res


if __name__ == '__main__':
    lncrna_dict = read_fa("miRBaseSequence.fasta") 
    pool = Pool(cpu_count())
    print("miRNA-doc2vec")
    lncrna_doc2vecs = pool.map(lncrna_doc2vec_embedding,list(lncrna_dict.values()))
    lnc_doc2vec_dict = to_dict(lncrna_dict,lncrna_doc2vecs)
    save_dict(lnc_doc2vec_dict, "miRNA_doc2vec_dict_total.csv") 

