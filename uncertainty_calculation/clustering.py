import numpy as np
from collections import defaultdict
import yaml
import os

from uncertainty_calculation.inference import Inference
from uncertainty_calculation.addon.matching import matching

class Clustering:
    def __init__(self, _config):
        self._clustering_output = _config['clustering_config']['cluster_output']

        self._sampling_technique = _config['sampling_technique']

        self._inference_output = _config['inference_config']['inference_output']
        self._approach = _config['inference_config']['approach']
        self._fpass = _config['inference_config']['fpass']

        if self._sampling_technique == 'mcd':
            self._pu_output = self._inference_output + 'MCD/pixel_pred/' 
        elif self._sampling_technique == 'ensemble':
            self._pu_output = self._inference_output + 'Ensemble/pixel_pred/'

        self.infer = Inference(_config)

    def cluster(self):
        
        Y_pred = self.infer.insts_loader()
        
        Y_pred = np.array(Y_pred, dtype=np.int32)
        if not os.path.exists(self._clustering_output):
            os.makedirs(self._clustering_output)
        num_imgs = len(Y_pred[0])
        model_iteration = len(Y_pred)
        clusters = {}

        for img in range(num_imgs):
            
            graph = {}
            for j in range(model_iteration):

                dd = defaultdict(list)
                d1 ={}
                for i in range(model_iteration):
                    m = matching(Y_pred[j][img], Y_pred[i][img], report_matches=True)
                    no_match = False
                    try:
                        true_ind = m.true_ind+1
                        pred_ind = m.pred_ind+1
                    except:
                        no_match = True
                    d2 = {}
                    [d2.setdefault(i, -1) for i in range(1, m.n_true+1)]
                    if not no_match:
                        for x in true_ind:
                            d2[x] = int(pred_ind[np.where(true_ind==x)[0][0]])
                    for d in (d1, d2): 
                        for key, value in d.items():
                            dd[key].append(value)
                graph[j] = dd

            intersect = {}
            noninter = {}
            for model_0 in range(len(graph)):
                inter = {}
                nonint = {}
                for model_1 in range(len(graph)):
                    if model_0!=model_1:
                        for inst in graph[model_0]:
                            if model_0>0:
                                for i in range(model_0):
                                    if graph[model_0][inst] not in graph[model_1].values() and graph[model_0][inst] not in intersect[i].values():
                                        nonint[inst] = graph[model_0][inst]

                            else:
                                if graph[model_0][inst] in graph[model_1].values():
                                    inter[inst] = graph[model_0][inst]

                                elif graph[model_0][inst] not in graph[model_1].values():
                                    nonint[inst] = graph[model_0][inst]

                intersect[model_0] = inter
                noninter[model_0] = nonint


            for model_0 in intersect:
                for inst in intersect[model_0]:
                    for model_1 in noninter:
                        if intersect[model_0][inst] in noninter[model_1].values():
                            
                            noninter[model_1].pop(list(noninter[model_1].keys())[list(noninter[model_1].values()).index(intersect[model_0][inst])])

            for model_0 in noninter:
                for inst in noninter[model_0]:
                    for model_1 in noninter:
                        if model_0!=model_1:
                            if noninter[model_0][inst] in noninter[model_1].values():
                            
                                noninter[model_1].pop(list(noninter[model_1].keys())[list(noninter[model_1].values()).index(noninter[model_0][inst])])

            if len(intersect)==len(noninter):
                for key in intersect:
                    if not noninter[key]:
                        noninter.pop(key)
            else:
                print('Error in lenth of dict')

            for model in noninter:
                for key in noninter[model]:
                    intersect[model][key] = noninter[model][key]

            clusters[img] = intersect

            if img == 0:
                print(clusters)

        with open(self._clustering_output + 'clusters.yml', 'w') as yaml_file:
            yaml.dump(clusters, yaml_file, default_flow_style=False)
        return


    def load_cluster(self):
        with open(self._clustering_output + 'clusters.yml', 'r') as yaml_file:
            return yaml.safe_load(yaml_file)