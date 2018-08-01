import pandas as pd
import numpy as np
from scipy.sparse import lil_matrix
from rescal import rescal_als
import logging
entitiy_index = {}
rel_index = {}
def read_data(filename):
    """
    Read relation / entities in freebase format.
    Return the corresponding tensor.
    X[i,:,:] = adjacent matrix for relation R_i.
    """
    rel_mat = list()
    data_table = pd.read_table(filename, header=None)
    entity = pd.concat([data_table[0], data_table[2]])
    entity = entity.unique()
    assert(len(entity) == 14951)
    rel = data_table[1].unqiue()
    assert(len(rel) == 1345)

    entity_n = len(entity)

    entity_index = {k:v for k,v in enumerate(entity)}


    data_table.columns = ["head", "rel", "tail"]
    # Packing it into a three-indices tensor.
    print("Packaging Tensors...")
    for r in rel:
        tensor_slice = np.zeros([entity_n, entity_n])
        rel_dataFrame = data_table[data_table["rel"].str.contains(r)]
        for idx, row in rel_dataFrame.iterrows():
            head_index = entity_index[row[0]]
            tail_index = entity_index[row[2]]
            tensor_slice[head_index, tail_index] = 1
        tensor_slice = lil_matrix(tensor_slice)
        rel_mat.append(tensor_slice)
    return rel_mat
if __name__ == "__main__":
    X = read_data("./FB15k/freebase_mtr100_mte100-train.txt")
    print("Training starts.")
    A, R, fit, itr, exectimes = rescal_als(X, 100, init='nvecs', lambda_A=10, lambda_R=10)