import yaml
import matplotlib.pyplot as plt
from argparse import ArgumentParser

from uncertainty_calculation.addon.plot import calibration_plot

class Plot:

    def __init__(self, _config):

        self.spatial = False
        self.fraction = False
        self.hybrid = False
        
        if _config['calibration_plot_config']['approach'] == 'all' or _config['calibration_plot_config']['approach'] == 'spatial':
            self.spatial = True
        if _config['calibration_plot_config']['approach'] == 'all' or _config['calibration_plot_config']['approach'] == 'fraction':
            self.fraction = True
        if _config['calibration_plot_config']['approach'] == 'all' or _config['calibration_plot_config']['approach'] == 'hybrid':
            self.hybrid = True

        self.calibration_output = _config['calibration_config']['calibration_output']
        self.cali_plot = _config['calibration_plot_config']['calibration_plot_output']
        self.bin = _config['calibration_plot_config']['bin']

    def new_bin_size(self, tp):
        new_bin_size = {}
        k = 1
        for i in tp:
            if k==1:
                j = i
                k+=1
                continue
            if k%2==0:
                new_bin_size[i] = tp[i]+tp[j]
            j = i
            k+=1

        return new_bin_size

    def uncertainty_type(self):

        if self.fraction:
            typ = 'frac_'
        elif self.spatial:
            typ = 'spl_'
        elif self.hybrid:
            typ = 'hyb_'
        else:
            raise ValueError("You have to specify which uncertainty to calculate (e.g. approach: 'hybrid).")

        return typ


    def calibration(self):

        typ = self.uncertainty_type()

        with open(self.calibration_output + typ + 'fp.yml', 'r') as yaml_file:
            fp =  yaml.safe_load(yaml_file)
        with open(self.calibration_output + typ + 'tp.yml', 'r') as yaml_file:
            tp =  yaml.safe_load(yaml_file)
        if self.bin:
            calibration_plot(self.new_bin_size(tp), self.new_bin_size(fp), spl=self.spatial, frac=self.fraction, hyb=self.hybrid)
        else:
            calibration_plot(tp, fp, spl=self.spatial, frac=self.fraction, hyb=self.hybrid)

        if self.spatial:
            plt.savefig(self.cali_plot + '_spl.png', bbox_inches='tight')
        elif self.fraction:
            plt.savefig(self.cali_plot + '_frac.png', bbox_inches='tight')
        elif self.hybrid:
            plt.savefig(self.cali_plot + '_hyb.png', bbox_inches='tight')

        return


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("config", help="yaml config", type=str)
    args = parser.parse_args()

    config_file = getattr(args, 'config', False)

    with open(config_file,encoding='utf8') as f:
        config = yaml.safe_load(f)

    plot = Plot(config)
    plot.calibration()