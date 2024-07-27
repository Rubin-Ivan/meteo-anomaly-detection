import numpy as np
import pandas as pd
from bamt.networks import HybridBN

def anomaly_detection(data,bn_name,month,threshold, land_mask=None, clusters=None):
    
    """
    Detect anomalies in the given data using a Bayesian Network and return the probability masks.

    Args:
        data (np.ndarray): A 2D np array corresponding to one time value.
        bn_name (str): The name of the Bayesian Network file to load (for a specific sea).
        month (int): The month that the data corresponds to.
        threshold (float): The probability threshold for determining anomalies.
        land_mask (np.ndarray, optional): The land mask array.
        clusters (np.ndarray, optional): The clusters array. 
    Returns:

        prob_mask (np.ndarray): The probability mask.
        prob_mask_int (np.ndarray): The binary mask indicating anomalies (1) and normal points (0).
    """
    
    list_znach=[]
    list_month=[]
    list_cluster=[]
    
    list_X=[]
    list_Y=[]

    for y in range(data.shape[0]):
                for x in range(data.shape[1]):
                    if (land_mask is not None):
                        if land_mask[y,x] == 0:

                            list_znach.append(data[y,x])
                            list_Y.append(y)
                            list_X.append(x)
                            list_month.append(month)

                            if (clusters is not None):
                                list_cluster.append(clusters[y,x])
                            else:
                                list_cluster.append(2)
                    else:
                        list_znach.append(data[y,x])
                        list_Y.append(y)
                        list_X.append(x)
                        list_month.append(month)

                        if (clusters is not None):
                            list_cluster.append(clusters[y,x])
                        else:
                            list_cluster.append(1)

                        
    df = pd.DataFrame({
    'VALUE':list_znach ,
    'TIME': list_month,
    'CLUSTER':list_cluster,
    'x':list_X,
    'y':list_Y
    })
    
    df['CLUSTER'] = df['CLUSTER'].astype(int)
    df['TIME'] = df['TIME'].astype(int)
    df['CLUSTER'] = df['CLUSTER'].astype(str)
    df['TIME'] = df['TIME'].astype(str)
    df=df.round(2)
    df.loc[df['CLUSTER'] == '0', 'CLUSTER'] = '3'
    data_test2=df.drop(columns=['x','y'])

    bn = HybridBN(has_logit=True)
    bn.load(bn_name)
    prob=probability_list(data_test2, bn, ['VALUE', 'CLUSTER', 'TIME'], 'NORM')

    df['prob']=prob
    
    max_x = df['x'].max()
    max_y = df['y'].max()
    prob_mask = np.zeros((max_y + 1, max_x + 1))
    for _, row in df.iterrows():
        prob_mask[row['y'], row['x']] = row['prob']
        
    prob_mask_int = (prob_mask >= threshold).astype(int)

    return prob_mask, prob_mask_int
    
def probability_list(data, bn, node_list, node_class):

    probability_list=[]

    for i in range(data.shape[0]):       

        probability_list.append(bn.get_dist(node_class, {node_list[0]:data[node_list[0]][i], 
                                                         node_list[1]:data[node_list[1]][i],
                                                         node_list[2]:data[node_list[2]][i]})[0])

    return probability_list

if __name__ == "__main__":

    data = np.load('data_Чукотское.npy')
    land_mask = np.load('mask_Чукотское.npy')
    clusters = np.load('clusters_Чукотское.npy')
    threshold = 0.2
    month = 1
    bn_name = 'Чукотское_BN.json'


    prob_mask, anomaly_mask = anomaly_detection(data[0], bn_name, month, threshold, land_mask, clusters)


