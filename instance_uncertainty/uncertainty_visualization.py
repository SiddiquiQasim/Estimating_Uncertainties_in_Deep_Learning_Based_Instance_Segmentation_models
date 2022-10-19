import matplotlib.pyplot as plt
import numpy as np
import cv2
from argparse import ArgumentParser
import yaml

from uncertainty_calculation.DataLoader import DataLoader
from uncertainty_calculation.radial_uncertainty import Radial_Uncertainty
from uncertainty_calculation.pixel_uncertainty import Pixel_Uncertainty
from uncertainty_calculation.clustering import Clustering
from uncertainty_calculation.addon.plot import draw_polygons
from stardist.geometry import _polygons_to_label_old as polygons_to_label

class Pixel_Uncertainty_Visualization:

    def __init__(self, _config):
        self.pixel_uncertainty = Pixel_Uncertainty(_config)
        clustering = Clustering(_config)
        self._clusters = clustering.load_cluster()

        _data_set = _config['inference_config']['data_set']
        self._data_loader = DataLoader(_config)
        self._X_val = self._data_loader.data_loader(type=_data_set, normalize = True)[0]

        self.visual_output = _config['visualize_config']['visual_output']

    def single_instance_uncertainty(self, img, model, inst):

        mean_pred = self.pixel_uncertainty.mean_pred(img, model, inst, mean_threshold=0.5)
        mean = np.copy(mean_pred[0][img])
        Y_min = np.copy(mean_pred[0][img])
        Y_max = np.copy(mean_pred[0][img])
        Y_min[Y_min != np.max(np.unique(mean_pred[0][img]))] = 0
        Y_min[Y_min == np.max(np.unique(mean_pred[0][img]))] = 1
        Y_max[Y_max != np.min(np.unique(mean_pred[0][img]))] = 1
        

        poly_min, _ = cv2.findContours(np.array(Y_min, dtype=np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        poly_max, _ = cv2.findContours(np.array(Y_max, dtype=np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        mean[mean>0.5] = 1
        mean[mean<=0.5] = 0

        poly, _ = cv2.findContours(np.array(mean, dtype=np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        try:
            x = poly[0][:, 0, 0]
            x = np.append(x, poly[0][:, 0, 0][:1])
            y = poly[0][:, 0, 1]
            y = np.append(y, poly[0][:, 0, 1][:1])
            minx = poly_min[0][:, 0, 0]
            minx = np.append(minx, poly_min[0][:, 0, 0][:1])
            miny = poly_min[0][:, 0, 1]
            miny = np.append(miny, poly_min[0][:, 0, 1][:1])
            maxx = poly_max[0][:, 0, 0]
            maxx = np.append(maxx, poly_max[0][:, 0, 0][:1])
            maxy = poly_max[0][:, 0, 1]
            maxy = np.append(maxy, poly_max[0][:, 0, 1][:1])
        except:
            x = []
            y = []
            minx = []
            miny = []
            maxx = []
            maxy = []

        return x, y, minx, miny, maxx, maxy

    def print_score(self, img, model, inst):

        spl_score = self.pixel_uncertainty.spl_uncertainty(self, img, model, inst, mean_threshold=0.5)
        frac_score = self.pixel_uncertainty.fraction_uncertainty(self, img, model, inst)
        hyb_score = spl_score * frac_score

        return hyb_score


    def visualize(self, img):
        xaxis = []
        yaxis = []
        score = []
        min_xaxis = []
        min_yaxis = []
        max_xaxis = []
        max_yaxis = []
        for model in self._clusters[img]:
            if model > 0:
                break
            for inst in self._clusters[img][model]:
                
                print('img-{}_mode-{}_inst-{}'.format(img, model,inst))
                sc = self.print_score(img, model, inst)
                x, y, minx, miny, maxx, maxy = self.single_instance_uncertainty(img, model, inst)
                xaxis.append(x)
                yaxis.append(y)
                score.append(sc)
                min_xaxis.append(minx)
                min_yaxis.append(miny)
                max_xaxis.append(maxx)
                max_yaxis.append(maxy)
                
        plt.figure(figsize=(13,6))
        plt.imshow(self._X_val[img], cmap='gray', clim=(0,1))
        for j in range(len(xaxis)):
            try:
                plt.plot(xaxis[j] , yaxis[j], '--', alpha=1, linewidth=3, zorder=1, color='red', label='Prediction')
                plt.plot(min_xaxis[j] , min_yaxis[j], '--', alpha=1, linewidth=1.5, zorder=1, color='yellow', label='Prediction')
                plt.plot(max_xaxis[j] , max_yaxis[j], '--', alpha=1, linewidth=1.5, zorder=1, color='yellow', label='Prediction')
                plt.text(max(xaxis[j]), max(yaxis[j]), '{:.2f}'.format(score[j]),color='red', fontsize=14)
            except:
                continue
        plt.axis('off')
        plt.savefig(self.visual_output, bbox_inches="tight")

class Radial_Uncertainty_Visualization:

    def __init__(self, _config):

        self.radial_uncertainty = Radial_Uncertainty(_config)

        _data_set = _config['inference_config']['data_set']
        self._data_loader = DataLoader(_config)
        self._X_val = self._data_loader.data_loader(type=_data_set, normalize = True)[0]

        self.visual_output = _config['uncertainty_visual']['visual_output']

    def print_score(self, img, inst, mean_prob, points, median_coord):
        
        median_label = polygons_to_label(median_coord, mean_prob[0][img], np.reshape(points[inst], (1,2)), thr=0.5)
        spl_score = self.radial_uncertainty.spl_uncertainty(img, inst, points, median_label)
        frac_score = self.radial_uncertainty.fraction_uncertainty(img, inst, points)
        wiou_score = spl_score * frac_score
        return wiou_score

    def visualize(self, img):

        mean_prob, mean_dist = self.radial_uncertainty.calculate_mean()
        points = self.radial_uncertainty.predict_center(img, mean_prob, mean_dist)
        min_coord, max_coord = self.radial_uncertainty.predict_boundary(img)
        median_coord = self.radial_uncertainty.predict_median(img)

        plt.figure(figsize=(13,6))
        plt.imshow(self._X_val[img],cmap='gray')
        x_max, y_max = draw_polygons(max_coord,mean_prob[0][img],points,show_dist=False)
        x_min, y_min = draw_polygons(min_coord,mean_prob[0][img],points,show_dist=False)
        draw_polygons(median_coord,mean_prob[0][img],points,show_dist=True,median=True)     
        for i in range(len(x_max)):
            plt.plot(x_max[i],y_max[i],'--', alpha=1, linewidth=1.5, color='yellow')
            plt.plot(x_min[i],y_min[i],'--', alpha=1, linewidth=1.5, color='yellow')
            plt.text(max(x_max[i]), max(y_max[i]), '{:.2f}'.format(self.print_score(img, i, mean_prob, points, median_coord)),color='red', fontsize=14)
            for j in range((len(x_max[i]))):
                plt.plot([x_max[i][j], x_min[i][j]], [y_max[i][j], y_min[i][j]],'--', alpha=1, linewidth=1, color='yellow')
        plt.axis('off')
        plt.savefig(self.visual_output, bbox_inches="tight")

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("config", help="yaml config", type=str)
    parser.add_argument("img_ind", help="Image Index", type=int)
    args = parser.parse_args()

    config_file = getattr(args, 'config', False)
    img = getattr(args, 'img_ind', 0)

    with open(config_file,encoding='utf8') as f:
        config = yaml.safe_load(f)

    if config['visualize_config']['approach'] == 'pixel_uncertainty':
        unc = Pixel_Uncertainty_Visualization(config)
        unc.visualize(img)

    if config['visualize_config']['approach'] == 'radial_uncertainty':
        unc = Radial_Uncertainty_Visualization(config)
        unc.visualize(img)