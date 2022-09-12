import os
import cv2
from tqdm import tqdm
import yaml

from stardist import fill_label_holes
from csbdeep.utils import normalize


class DataLoader:

    def __init__(self, _config):
        self._data_path = _config['spliting_config']['data_path']
        self._split_path = _config['spliting_config']['split_path']
        self._data_split = _config['spliting_config']['data_split']
        

    def data_split(self):
        '''
        Read Data
        '''
        
        if not os.path.exists(self._split_path):
            os.makedirs(self._split_path)

        X_ind = os.listdir(self._data_path + 'img/')
        for i in X_ind:
            if not i.endswith('.png'):
                X_ind.remove(i)
        X_ind = sorted(X_ind)

        Y_ind = os.listdir(self._data_path + 'mask/')
        for i in Y_ind:
            if not i.endswith('.png'):
                Y_ind.remove(i)
        Y_ind = sorted(Y_ind)

        assert X_ind==Y_ind, 'Number of images and masks do not match'

        '''
        Split into train and validation datasets.
        '''
        assert len(X_ind) > 1, "not enough training data"

        n_val = max(1, int(round(self._data_split[1] * len(X_ind))))
        X_val_ind, Y_val_ind = [X_ind[i] for i in range(len(X_ind)-n_val, len(X_ind))]  , [Y_ind[i] for i in range(len(X_ind)-n_val, len(X_ind))]
        X_trn_ind, Y_trn_ind = [X_ind[i] for i in range(len(X_ind)-n_val)], [Y_ind[i] for i in range(len(X_ind)-n_val)] 
        print('number of images: %3d' % len(X_ind))
        print('- training:       %3d' % len(X_trn_ind))
        print('- validation:     %3d' % len(X_val_ind))
        
        data_conf = {}
        data_conf['total'] = len(X_ind)
        data_conf['training'] = len(X_trn_ind)
        data_conf['validation'] = len(X_val_ind)
        data_conf['X_trn_ind'] = X_trn_ind
        data_conf['Y_trn_ind'] = Y_trn_ind
        data_conf['X_val_ind'] = X_val_ind
        data_conf['Y_val_ind'] = Y_val_ind
        
        with open(self._split_path + 'data_conf.yml', 'w') as yaml_file:
            yaml.dump(data_conf, yaml_file, default_flow_style=False)
        return



    def normalize_img(self, X, Y):
        '''
        Normalize images and fill small label holes.
        '''

        axis_norm = (0,1)   # normalize channels independently
        X = [normalize(x,1,99.8,axis=axis_norm) for x in tqdm(X)]
        Y = [fill_label_holes(y) for y in tqdm(Y)]

        return X, Y

    def data_loader(self, type='test', normalize=False):
    
        X_ind, Y_ind = self.data_type(type=type)
        assert set(X_ind) == set(Y_ind) 
        X = []
        Y = []
        for i in tqdm(X_ind):

            x = cv2.imread(self._data_path + 'img/'+ i, 0)
            X.append(x)
            y = cv2.imread(self._data_path + 'mask/'+ i, 0)
            Y.append(y)
        print("Images loaded")
        if normalize:
            return self.normalize_img(X, Y)
        else:
            return X, Y


    def data_type(self, type='test'):
        with open(self._split_path + 'data_conf.yml', 'r') as yaml_file:    
            data_conf = yaml.safe_load(yaml_file)

        
        if type == 'train':
            print("Loading training images ....")
            X_ind = data_conf['X_trn_ind']
            Y_ind = data_conf['Y_trn_ind']
        elif type == 'test':
            print("Loading testing images ....")
            X_ind = data_conf['X_val_ind']
            Y_ind = data_conf['Y_val_ind']
        
        return X_ind, Y_ind