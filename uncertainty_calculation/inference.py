import numpy as np
from tqdm import tqdm
import os

from uncertainty_calculation.DataLoader import DataLoader
from uncertainty_calculation.addon.model import StarDist2D_Unc

class Inference:

    def __init__(self, _config):
        self._sampling_technique = _config['sampling_technique']

        self._model_path = _config['training_config']['model_path']
        self._n_rays = _config['training_config']['n_rays']
        self._dropout_rate = _config['training_config']['dropout_rate']
        self._mcd_pos = _config['training_config']['mcd_pos']
        if self._sampling_technique == 'ensemble':
            self._num_models = _config['training_config']['num_models']
 
        self._data_loader = DataLoader(_config)

        self._inference_output = _config['inference_config']['inference_output']
        self._approach = _config['inference_config']['approach']
        self._data_set = _config['inference_config']['data_set']
        self._fpass = _config['inference_config']['fpass']

        if self._sampling_technique == 'mcd':
            self._model_path = self._model_path + 'MCD'
            self._pu_output = self._inference_output + 'MCD/pixel_pred/' 
            self._ru_output = self._inference_output + 'MCD/radial_pred/' 
        elif self._sampling_technique == 'ensemble':
            self._model_path = self._model_path + 'Ensemble_'
            self._pu_output = self._inference_output + 'Ensemble/pixel_pred/' 
            self._ru_output = self._inference_output + 'Ensemble/radial_pred/' 
        ### TODO per_output and train_data


    def inference(self):
        if self._sampling_technique == 'ensemble':
            if self._fpass > self._num_models:
                raise ValueError("number of  fpass is out of bound (fpass <= num_models).")
        
        X_val = np.array(self._data_loader.data_loader(type=self._data_set, normalize = True)[0], dtype=np.float32)
        
        print('Making Prediction...')
        ints = np.zeros((self._fpass, X_val.shape[0], X_val.shape[1], X_val.shape[2]), dtype=np.int32)
        
        if self._sampling_technique == 'mcd':
            model = StarDist2D_Unc(None, name='stardist', basedir=self._model_path, dropout_rate=self._dropout_rate, mcd_pos=self._mcd_pos)
            for idx in tqdm(range(self._fpass)):
                dst = np.zeros((X_val.shape[0], X_val.shape[1], X_val.shape[2], self._n_rays), dtype=np.float32)
                prb = np.zeros((X_val.shape[0], X_val.shape[1], X_val.shape[2]), dtype=np.float32)
                for x in range(len(X_val)):
                    prb[x], dst[x] = model.predict(X_val[x])
                    ints[idx][x], _ = model.predict_instances(X_val[x], n_tiles=model._guess_n_tiles(X_val[x]), show_tile_progress=False)
                
                if self._approach == 'all' or self._approach == 'radial_uncertainty':
                    if not os.path.exists(self._ru_output):
                        os.makedirs(self._ru_output)
                    np.save(self._ru_output + 'dst_m-{}.npy'.format(idx), dst)
                    np.save(self._ru_output + 'prb_m-{}.npy'.format(idx), prb)



        elif self._sampling_technique == 'ensemble':
            for idx in tqdm(range(self._fpass)):
                model = StarDist2D_Unc(None, name='stardist', basedir=self._model_path+'{}'.format(idx), dropout_rate=self._dropout_rate, mcd_pos=self._mcd_pos)
                dst = np.zeros((X_val.shape[0], X_val.shape[1], X_val.shape[2], self._n_rays), dtype=np.float32)
                prb = np.zeros((X_val.shape[0], X_val.shape[1], X_val.shape[2]), dtype=np.float32)
                for x in range(len(X_val)):
                    prb[x], dst[x] = model.predict(X_val[x])
                    ints[idx][x], _ = model.predict_instances(X_val[x], n_tiles=model._guess_n_tiles(X_val[x]), show_tile_progress=False)
                    
                if self._approach == 'all' or self._approach == 'radial_uncertainty':
                    if not os.path.exists(self._ru_output):
                        os.makedirs(self._ru_output)
                    np.save(self._ru_output + 'dst_m-{}.npy'.format(idx), dst)
                    np.save(self._ru_output + 'prb_m-{}.npy'.format(idx), prb)
            
                
        print('Saving Prediction')
        if self._approach == 'all' or self._approach == 'pixel_uncertainty':
            if not os.path.exists(self._pu_output):
                os.makedirs(self._pu_output)
            np.save(self._pu_output + 'insts_fpass-{}.npy'.format(self._fpass), ints)

        
        return

    def raw_data_loader_boot(self, _count, num_img, img_shape, n_ray, bootstrap):
        
        raw_prob = np.zeros((_count, num_img, img_shape[0], img_shape[1]), dtype=np.float32)    
        raw_dist = np.zeros((_count, num_img, img_shape[0], img_shape[1], n_ray), dtype=np.float32)
        
        k = 0
        if not bootstrap:
            np.random.seed(10)
        rand = np.random.permutation(self._fpass)[:_count]
        for i in rand:
            raw_prob[k] = np.load(self._ru_output + 'prb_m-{}.npy'.format(i), encoding='latin1', allow_pickle=True) 
            raw_dist[k] = np.load(self._ru_output + 'dst_m-{}.npy'.format(i), encoding='latin1', allow_pickle=True)
            k+=1
        
        return raw_prob, raw_dist

    def insts_loader(self):
        return np.load(self._pu_output + 'insts_fpass-{}.npy'.format(self._fpass))