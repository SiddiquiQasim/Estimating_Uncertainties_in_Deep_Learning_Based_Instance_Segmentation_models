from uncertainty_calculation.pixel_uncertainty import Pixel_Uncertainty
from uncertainty_calculation.radial_uncertainty import Radial_Uncertainty


class Calibration:
    def __init__(self, _config):
        self._config = _config
        self._approach = _config['calibration_config']['approach']
        if _config['calibration_config']['bootstrap']:
            self._iteration = _config['calibration_config']['iteration']
        else:
            self._iteration = 1
            

    def cali_pixel_unc(self):

        for idx in range(self._iteration):
            pixel = Pixel_Uncertainty(self._config)
            pixel.per_bin_all(idx)


    def cali_radial_unc(self):
                
        for idx in range(self._iteration):
            radial = Radial_Uncertainty(self._config)
            radial.per_bin_all(idx)

    def calibrate(self):
        if self._approach == 'pixel_uncertainty':
            self.cali_pixel_unc()
        elif self._approach == 'radial_uncertainty':
            self.cali_radial_unc()