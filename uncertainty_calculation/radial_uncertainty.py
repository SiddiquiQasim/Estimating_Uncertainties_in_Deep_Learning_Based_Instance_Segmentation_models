import os
import numpy as np

from uncertainty_calculation.addon.matching import matching
from uncertainty_calculation.addon.save_cali_dic import tp_fp_save_all, save_score_dic

from uncertainty_calculation.inference import Inference
from uncertainty_calculation.DataLoader import DataLoader

from stardist.geometry import _dist_to_coord_old as dist_to_coord
from stardist.geometry import _polygons_to_label_old as polygons_to_label
from stardist.nms import _non_maximum_suppression_old as non_maximum_suppression



class Radial_Uncertainty:

    def __init__(self, _config):

        _data_set = _config['inference_config']['data_set']
        self._data_loader = DataLoader(_config)
        self._Y_true = self._data_loader.data_loader(type=_data_set, normalize = True)[1]

        self._n_rays = _config['training_config']['n_rays']

        self._n_bin = _config['calibration_config']['n_bin']
        self._count = _config['calibration_config']['count']
        self._save_interval = _config['calibration_config']['save_interval']
        self._save_score = _config['calibration_config']['save_score']
        self._calibration_output = _config['calibration_config']['calibration_output'] + 'radial_uncertainty/'
        self._bootstrap = _config['calibration_config']['bootstrap']
        if not os.path.exists(self._calibration_output):
            os.makedirs(self._calibration_output)
        
        infer = Inference(_config)
        self._raw_prob, self._raw_dist = infer.raw_data_loader_boot(self._count, num_img=len(self._Y_true),
                                                                img_shape=(self._Y_true[0].shape[0],self._Y_true[0].shape[1]), n_ray=self._n_rays,
                                                                 bootstrap = self._bootstrap)


    def calculate_mean(self):
        
        mean_prob = np.mean(self._raw_prob, axis=0, keepdims=True)
        mean_dist = np.mean(self._raw_dist, axis=0, keepdims=True)

        return mean_prob, mean_dist

    def predict_center(self, img_ind, mean_prob, mean_dist):

        coord = dist_to_coord(mean_dist[0][img_ind])
        points = non_maximum_suppression(coord, mean_prob[0][img_ind], prob_thresh=0.5)

        return points

    def predict_median(self, img_ind):
        
        dist_q500 = np.percentile(self._raw_dist[:,img_ind,...], 50, axis=0, keepdims=True)
        median_coord = dist_to_coord(dist_q500[0])

        return median_coord

    def predict_boundary(self, img_ind):
  
        dist_q025, dist_q975 = np.percentile(self._raw_dist[:,img_ind,...], [2.5,97.5], axis=0, keepdims=True)
        max_coord = dist_to_coord(dist_q975[0])
        min_coord = dist_to_coord(dist_q025[0])

        return min_coord, max_coord


    def spl_uncertainty(self, img, inst, points, median_label):
        spls = []
        for model in range(self._count):
            label = polygons_to_label(dist_to_coord(self._raw_dist[model][img]), self._raw_prob[model][img], np.reshape(points[inst], (1,2)), thr=0.5)
            assert label.shape == self._raw_prob[model][img].shape
            m = matching(label, median_label, report_matches=True)
            spls.append(m.mean_matched_score)

                
        return np.mean(spls)

    def fraction_uncertainty(self, img, inst, points):
        num = self._count
        for model in range(self._count):
            label = polygons_to_label(dist_to_coord(self._raw_dist[model][img]), self._raw_prob[model][img], np.reshape(points[inst], (1,2)), thr=0.5)
            assert label.shape == self._raw_prob[model][img].shape            
            if set(np.unique(label)) == set(np.array([0])):
                print('fraction uncertainty #########################################################')
                num-=1
        return (num/self._count)

    def true_positive(self, img, median_label, Y_true):
        
        m = matching(Y_true[img], median_label, report_matches=True)
        if m.tp == 1:
            return True
        elif m.tp > 1:
            raise ValueError
        else:
            return False

    def per_bin_all(self, idx):
        new_path = self._calibration_output[:-1]+'itr_{}/'.format(idx)
        if not os.path.exists(new_path):
            os.makedirs(new_path)
        k = 0
        a = 0
        b = 0
        c = 0
        d = 0
        e = 0
        f = 0
        print('k = 0')
        unc = np.linspace(0,1,self._n_bin+1) # [0, 0.05, 0.1, 0.15, ..., 0.95, 1.00]
        tp_spl = {}
        fp_spl = {}
        tp_frac = {}
        fp_frac = {}
        tp_hyb = {}
        fp_hyb = {}
        for i in unc:
            tp_spl[str(i)] = 0
            fp_spl[str(i)] = 0
            tp_frac[str(i)] = 0
            fp_frac[str(i)] = 0
            tp_hyb[str(i)] = 0
            fp_hyb[str(i)] = 0
        
        spl_score_dic = {}
        frac_score_dic = {}
        hyb_score_dic = {}

        # raw_prob, raw_dist = raw_prob[:count, ...], raw_dist[:count, ...]
        mean_prob, mean_dist = self.calculate_mean()
        for img in range(len(self._raw_dist[0])):
            points = self.predict_center(img, mean_prob, mean_dist)
            median_coord = self.predict_median(img)
            for inst in range(len(points)):
                median_label = polygons_to_label(median_coord, mean_prob[0][img], np.reshape(points[inst], (1,2)), thr=0.5)
                if set([0]) == set(np.unique(median_label)):
                    print('################ Median is less than 0.5 ################')
                    continue
                spl_score = self.spl_uncertainty(img, inst, points, median_label)
                frac_score = self.fraction_uncertainty(img, inst, points)
                hyb_score = spl_score * frac_score
                print('{}, spl_score {}'.format(k, spl_score))
                print('{}, frac_score {}'.format(k, frac_score))
                print('{}, hyb_score {}'.format(k, hyb_score))
                
                for i in range(len(unc)-1):
                    if spl_score > unc[i] and spl_score <= unc[i+1]:
                        if self.true_positive(img, median_label, self._Y_true):
                            print('img-{}_inst-{} True Positive[{},{}] spl'.format(img, inst, unc[i], unc[i+1]))
                            tp_spl[str(unc[i])]+=1
                            if a == 0:
                                spl_score_dic['tp_min_spl_score'] = spl_score
                                spl_score_dic['tp_min_spl_loc'] = [img, inst]
                            if spl_score<spl_score_dic['tp_min_spl_score']:
                                spl_score_dic['tp_min_spl_score'] = spl_score
                                spl_score_dic['tp_min_spl_loc'] = [img, inst]
                            a+=1
                        else:
                            print('img-{}_inst-{} False Positive[{},{}] spl'.format(img, inst, unc[i], unc[i+1]))
                            fp_spl[str(unc[i])]+=1
                            if b == 0:
                                spl_score_dic['fp_max_spl_score'] = spl_score
                                spl_score_dic['fp_max_spl_loc'] = [img, inst]
                            if spl_score>spl_score_dic['fp_max_spl_score']:
                                spl_score_dic['fp_max_spl_score'] = spl_score
                                spl_score_dic['fp_max_spl_loc'] = [img, inst]
                            b+=1
                    if frac_score > unc[i] and frac_score <= unc[i+1]:
                        if self.true_positive(img, median_label, self._Y_true):
                            print('img-{}_inst-{} True Positive[{},{}] frac'.format(img, inst, unc[i], unc[i+1]))
                            tp_frac[str(unc[i])]+=1
                            if c == 0:
                                frac_score_dic['tp_min_frac_score'] = frac_score
                                frac_score_dic['tp_min_frac_loc'] = [img, inst]
                            if frac_score<frac_score_dic['tp_min_frac_score']:
                                frac_score_dic['tp_min_frac_score'] = frac_score
                                frac_score_dic['tp_min_frac_loc'] = [img, inst]
                            c+=1
                        else:
                            print('img-{}_inst-{} False Positive[{},{}] frac  ###############################################################'.format(img, inst, unc[i], unc[i+1]))
                            fp_frac[str(unc[i])]+=1
                            if d == 0:
                                frac_score_dic['fp_max_frac_score'] = frac_score
                                frac_score_dic['fp_max_frac_loc'] = [img, inst]
                            if frac_score>frac_score_dic['fp_max_frac_score']:
                                frac_score_dic['fp_max_frac_score'] = frac_score
                                frac_score_dic['fp_max_frac_loc'] = [img, inst]
                            d+=1
                    if hyb_score > unc[i] and hyb_score <= unc[i+1]:
                        if self.true_positive(img, median_label, self._Y_true):
                            print('img-{}_inst-{} True Positive[{},{}] hyb'.format(img, inst, unc[i], unc[i+1]))
                            tp_hyb[str(unc[i])]+=1
                            if e == 0:
                                hyb_score_dic['tp_min_hyb_score'] = hyb_score
                                hyb_score_dic['tp_min_hyb_loc'] = [img, inst]
                            if hyb_score<hyb_score_dic['tp_min_hyb_score']:
                                hyb_score_dic['tp_min_hyb_score'] = hyb_score
                                hyb_score_dic['tp_min_hyb_loc'] = [img, inst]
                            e+=1
                        else:
                            print('img-{}_inst-{} False Positive[{},{}] wion'.format(img, inst, unc[i], unc[i+1]))
                            fp_hyb[str(unc[i])]+=1
                            if f == 0:
                                hyb_score_dic['fp_max_hyb_score'] = hyb_score
                                hyb_score_dic['fp_max_hyb_loc'] = [img, inst]
                            if hyb_score>hyb_score_dic['fp_max_hyb_score']:
                                hyb_score_dic['fp_max_hyb_score'] = hyb_score
                                hyb_score_dic['fp_max_hyb_loc'] = [img, inst]
                            f+=1
                        
                info = {'num_insts': k, 'img': img, 'inst': inst}
                k+=1
                if k % self._save_interval == 0:                    
                    tp_fp_save_all(idx, tp_spl, fp_spl, tp_frac, fp_frac, tp_hyb, fp_hyb, info, new_path)

            if self._save_score:
                save_score_dic(idx, spl_score_dic, frac_score_dic, hyb_score_dic, new_path)

        return tp_spl, fp_spl, spl_score_dic, tp_frac, fp_frac, frac_score_dic, tp_hyb, fp_hyb, hyb_score_dic