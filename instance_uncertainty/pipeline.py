import yaml
from argparse import ArgumentParser
from uncertainty_calculation.DataLoader import DataLoader
from uncertainty_calculation.train import TrainStarDist
from uncertainty_calculation.inference import Inference
from uncertainty_calculation.clustering import Clustering
from uncertainty_calculation.calibration import Calibration


class StarDistPipe():
    """
    StarDistTrain creates a easy to use pipeline to calculate uncertainty

    Note: The config must be consistent, otherwise modules from the pipeline can fail

    Args:
        config (dict): Pipeline configuration
    """

    def __init__(self, config):

        self._config = config

    def _splitting(self):
        """
        Spilts data into training and test sets
        """
        dataloader = DataLoader(self._config)
        dataloader.data_split()    
        return

    def _training(self):
        """
        Build and starts the CNN training
        """
        training = TrainStarDist(self._config)
        training.train()
        return

    def _infering(self):
        """
        Samples predictions for a given models
        """
        infer = Inference(self._config)
        infer.inference()
        return

    def _clustering(self):
        """
        Clusteres instance from the sample prediction
        """
        
        if self._config['calibration_config']['approach'] == 'radial_uncertainty':
            return print('\n###No need to cluster for Radial Uncertainty approach###\n')
        elif self._config['calibration_config']['approach'] == 'pixel_uncertainty':
            clustering = Clustering(self._config)
            clustering.cluster()
            return

    def _calibrating(self):
        """
        Check calibration
        """
        calibration = Calibration(self._config)
        calibration.calibrate()
        return

    

    def start(self, splitting=False, training=False, infering=False, clustering=False, calibrating=False, all=False):
        """
        Launch the processing steps of the pipeline

        Args:
            splitting (bool): Enable downloading
            training (bool): Enable augmentation
            infering (bool): Enable training
        """

        if splitting or all:
            self._splitting()
        if training or all:
            self._training()
        if infering or all:
            self._infering()
        if clustering or all:
            self._clustering()
        if calibrating or all:
            self._calibrating()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("config", help="yaml config defining the training pipleline", type=str)
    parser.add_argument("--splitting", help="Run _splitting module",action='store_true')
    parser.add_argument("--training", help="Run _training module",action='store_true')
    parser.add_argument("--infering", help="Run _infering module",action='store_true')
    parser.add_argument("--clustering", help="Run _clustering module", action='store_true')
    parser.add_argument("--calibrating", help="Run _calibrating module",action='store_true')
    parser.add_argument("--all", help="Run _all module",action='store_true')
    args = parser.parse_args()

    if not any([args.splitting, args.training, args.infering,args.clustering, args.calibrating, args.all]):
        raise ValueError("You have to specify which module to run (e.g. --splitting).")

    should_split = getattr(args, 'splitting', False)
    should_train = getattr(args, 'training', False)
    should_infer = getattr(args, 'infering', False)
    should_cluster = getattr(args, 'clustering', False)
    should_calibrate = getattr(args, 'calibrating', False)
    should_all = getattr(args, 'all', False)

    config_file = getattr(args, 'config', False)

    with open(config_file,encoding='utf8') as f:
        config = yaml.safe_load(f)

    pipe = StarDistPipe(config)
    pipe.start(splitting=should_split, training=should_train, infering=should_infer,
                    clustering=should_cluster, calibrating=should_calibrate, all=should_all)