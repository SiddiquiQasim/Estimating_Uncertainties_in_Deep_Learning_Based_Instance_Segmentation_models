from uncertainty_calculation.DataLoader import DataLoader

from stardist import gputools_available
from stardist.models import Config2D
from uncertainty_calculation.addon.model import StarDist2D_Unc

class TrainStarDist:
    
    def __init__(self, _config):
        
        self._sampling_technique = _config['sampling_technique']
        self._model_path = _config['training_config']['model_path']
        self._n_rays = _config['training_config']['n_rays']
        self._epochs = _config['training_config']['epochs']
        
        self._dropout_rate = _config['training_config']['dropout_rate']
        self._mcd_pos = _config['training_config']['mcd_pos']
        if self._sampling_technique == 'ensemble':
            self._num_models = _config['training_config']['num_models']
        

        self._data_loader = DataLoader(_config)
        


    def train(self):

        X_trn, Y_trn = self._data_loader.data_loader(type='train', normalize = True)
        X_val, Y_val = self._data_loader.data_loader(type='test', normalize = True)

        grid = (1,1)
        use_gpu = True and gputools_available()
        n_channel = 1
        
        conf = Config2D (
            n_rays       = self._n_rays,
            grid         = grid,
            use_gpu      = use_gpu,
            n_channel_in = n_channel,
        )
        '''
        GPU usage
        '''

        if use_gpu:
            from csbdeep.utils.tf import limit_gpu_memory
            # adjust as necessary: limit GPU memory to be used by TensorFlow to leave some to OpenCL-based computations
            limit_gpu_memory(0.8)
            # alternatively, try this:
            # limit_gpu_memory(None, allow_growth=True)
            
        '''
        Dif Model with default values: prob_thresh=0.5, nms_thresh=0.4.
        '''

        if self._sampling_technique == 'mcd':
            model = StarDist2D_Unc(conf, name='stardist', basedir=self._model_path + 'MCD',
                                dropout_rate=self._dropout_rate, mcd_pos=self._mcd_pos)
            model.train(X_trn, Y_trn, validation_data=(X_val,Y_val), augmenter=None, epochs=self._epochs)
        elif self._sampling_technique == 'ensemble':
            for idx in range(self._num_models):
                model = StarDist2D_Unc(conf, name='stardist', basedir=self._model_path + 'Ensemble_{}'.format(idx),
                                    dropout_rate=self._dropout_rate, mcd_pos=self._mcd_pos)
                model.train(X_trn, Y_trn, validation_data=(X_val,Y_val), augmenter=None, epochs=self._epochs)
        else:
            raise ValueError("You have to specify sampling_technique as 'mcd' or 'ensemble'.")
            