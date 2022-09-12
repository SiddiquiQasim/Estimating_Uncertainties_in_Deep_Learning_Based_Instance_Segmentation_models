import os
import numpy as np

from uncertainty_calculation.clustering import Clustering
from uncertainty_calculation.DataLoader import DataLoader
from uncertainty_calculation.inference import Inference

from uncertainty_calculation.addon.matching import matching
from uncertainty_calculation.addon.save_cali_dic import tp_fp_save_all, save_score_dic


class Pixel_Uncertainty:
    ## TODO mean and iou threshold not hard coded

    def __init__(self, _config):

        _data_set = _config['inference_config']['data_set']
        self._data_loader = DataLoader(_config)
        self._Y_true = self._data_loader.data_loader(type=_data_set, normalize = True)[1]

        if _config['sampling_technique'] == 'mcd':
            self._inference_output = _config['inference_config']['inference_output'] + 'MCD/pixel_pred/'
        elif _config['sampling_technique'] == 'ensemble':
            self._inference_output = _config['inference_config']['inference_output'] + 'Ensemble/pixel_pred/'
        self._fpass = _config['inference_config']['fpass']      

        self._clustering_output = _config['clustering_config']['cluster_output']
        clustering = Clustering(_config)
        self._clusters = clustering.load_cluster()

        self._n_bin = _config['calibration_config']['n_bin']
        self._count = _config['calibration_config']['count']
        self._save_interval = _config['calibration_config']['save_interval']
        self._save_score = _config['calibration_config']['save_score']
        self._calibration_output = _config['calibration_config']['calibration_output'] + 'pixel_uncertainty/'
        self._bootstrap = _config['calibration_config']['bootstrap']

        
        infer = Inference(_config)

        self._Y_pred = infer.insts_loader()

        if not self._bootstrap:
            np.random.seed(10)
        self._rand = np.random.permutation(self._fpass)[:self._count]

    def spl_uncertainty(self, img, model, inst, mean_threshold=0.5): # Y_pred = (10,62,512,1024,32)
        
        # import pdb; pdb.set_trace()
        Y_pred_c = np.copy(self._Y_pred)
        m_pred = self.mean_pred(img, model, inst, mean_threshold=mean_threshold) # (512, 1024)
        spls = []
        assert  self._count <= len(Y_pred_c)
        for i in self._rand:
            if self._clusters[img][model][inst][i] != -1:
                m = matching(np.array(Y_pred_c[i,img,...], dtype=np.int32), np.array(m_pred, dtype=np.int32), report_matches=True)
                assert m.tp <= 1; 'Recheck code'
                spls.append(m.mean_matched_score)
        return np.mean(spls)
        
    def fraction_uncertainty(self, img, model, inst):
        inst_list = np.array([])
        for i in self._rand:
            inst_list = np.append(inst_list, self._clusters[img][model][inst][i])
        
        inst_list = inst_list[:self._count]
        inst_list[inst_list>0] = 1
        inst_list[inst_list<0] = 0
        return np.sum(inst_list) / self._count


    def mean_pred(self, img, model, inst, mean_threshold=0.5):
        
        Y_pred_c = np.copy(self._Y_pred)
        assert  self._count <= len(Y_pred_c)
        Y_premean = []
        for pred in self._rand:
            Y_pred_c[pred][img][Y_pred_c[pred][img]!=self._clusters[img][model][inst][pred]] = 0
            Y_pred_c[pred][img][Y_pred_c[pred][img]==self._clusters[img][model][inst][pred]] = 1
            if self._clusters[img][model][inst][pred] != -1:
                Y_premean.append(Y_pred_c[pred][img])
        
        mean_pred = np.mean(Y_premean, axis=0, keepdims=True)
        
        try:
            mean_pred = mean_pred[0]
            mean_pred[mean_pred>mean_threshold] = 1
            mean_pred[mean_pred<=mean_threshold] = 0
        except:
            mean_pred = Y_premean
        return mean_pred

    def true_positive(self, img, mean_pred):
        
        m = matching(self._Y_true[img], np.array(mean_pred, dtype=np.int32), report_matches=True)
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
        for img in self._clusters:
            z = 0
            for model in  self._clusters[img]:
                if z >= self._count:
                    break
                z+=1
                for inst in  self._clusters[img][model]:
                    # import pdb; pdb.set_trace()
                    m_pred = self.mean_pred(img, model, inst, mean_threshold=0.5) # (512,1024)
                    if len(m_pred) == 0:
                        continue
                    spl_score = self.spl_uncertainty(img, model, inst)
                    frac_score = self.fraction_uncertainty(img, model, inst)
                    hyb_score = spl_score * frac_score

                    print('{}, spl_score {}'.format(k, spl_score))
                    print('{}, frac_score {}'.format(k, frac_score))
                    print('{}, hyb_score {}'.format(k, hyb_score))

                    for i in range(len(unc)-1):
                        if spl_score > unc[i] and spl_score <= unc[i+1]:
                            if self.true_positive(img, m_pred):
                                print('img-{}_model-{}_inst-{} True Positive[{},{}] spl'.format(img, model, inst, unc[i], unc[i+1]))
                                tp_spl[str(unc[i])]+=1
                                if a == 0:
                                    spl_score_dic['tp_min_spl_score'] = spl_score
                                    spl_score_dic['tp_min_spl_loc'] = [img, model, inst]
                                if spl_score<spl_score_dic['tp_min_spl_score']:
                                    spl_score_dic['tp_min_spl_score'] = spl_score
                                    spl_score_dic['tp_min_spl_loc'] = [img, model, inst]
                                a+=1
                            else:
                                print('img-{}_model-{}_inst-{} False Positive[{},{}] spl'.format(img, model, inst, unc[i], unc[i+1]))
                                fp_spl[str(unc[i])]+=1
                                if b == 0:
                                    spl_score_dic['fp_max_spl_score'] = spl_score
                                    spl_score_dic['fp_max_spl_loc'] = [img, model, inst]
                                if spl_score>spl_score_dic['fp_max_spl_score']:
                                    spl_score_dic['fp_max_spl_score'] = spl_score
                                    spl_score_dic['fp_max_spl_loc'] = [img, model, inst]
                                b+=1

                        if frac_score > unc[i] and frac_score <= unc[i+1]:
                            if self.true_positive(img, m_pred):
                                print('img-{}_model-{}_inst-{} True Positive[{},{}] frac'.format(img, model, inst, unc[i], unc[i+1]))
                                tp_frac[str(unc[i])]+=1
                                if c == 0:
                                    frac_score_dic['tp_min_frac_score'] = frac_score
                                    frac_score_dic['tp_min_frac_loc'] = [img, model, inst]
                                if frac_score<frac_score_dic['tp_min_frac_score']:
                                    frac_score_dic['tp_min_frac_score'] = frac_score
                                    frac_score_dic['tp_min_frac_loc'] = [img, model, inst]
                                c+=1
                            else:
                                print('img-{}_model-{}_inst-{} False Positive[{},{}] frac'.format(img, model, inst, unc[i], unc[i+1]))
                                fp_frac[str(unc[i])]+=1
                                if d == 0:
                                    frac_score_dic['fp_max_frac_score'] = frac_score
                                    frac_score_dic['fp_max_frac_loc'] = [img, model, inst]
                                if frac_score>frac_score_dic['fp_max_frac_score']:
                                    frac_score_dic['fp_max_frac_score'] = frac_score
                                    frac_score_dic['fp_max_frac_loc'] = [img, model, inst]
                                d+=1

                        if hyb_score > unc[i] and hyb_score <= unc[i+1]:
                            if self.true_positive(img, m_pred):
                                print('img-{}_model-{}_inst-{} True Positive[{},{}] hyb'.format(img, model, inst, unc[i], unc[i+1]))
                                tp_hyb[str(unc[i])]+=1
                                if e == 0:
                                    hyb_score_dic['tp_min_hyb_score'] = hyb_score
                                    hyb_score_dic['tp_min_hyb_loc'] = [img, model, inst]
                                if hyb_score<hyb_score_dic['tp_min_hyb_score']:
                                    hyb_score_dic['tp_min_hyb_score'] = hyb_score
                                    hyb_score_dic['tp_min_hyb_loc'] = [img, model, inst]
                                e+=1
                            else:
                                print('img-{}_model-{}_inst-{} False Positive[{},{}] wion'.format(img, model, inst, unc[i], unc[i+1]))
                                fp_hyb[str(unc[i])]+=1
                                if f == 0:
                                    hyb_score_dic['fp_max_hyb_score'] = hyb_score
                                    hyb_score_dic['fp_max_hyb_loc'] = [img, model, inst]
                                if hyb_score>hyb_score_dic['fp_max_hyb_score']:
                                    hyb_score_dic['fp_max_hyb_score'] = hyb_score
                                    hyb_score_dic['fp_max_hyb_loc'] = [img, model, inst]
                                f+=1
                            
                    info = {'num_insts': k, 'img': img, 'model': model, 'inst': inst}
                    k+=1
                    if k % self._save_interval == 0:                    
                        tp_fp_save_all(idx, tp_spl, fp_spl, tp_frac, fp_frac, tp_hyb, fp_hyb, info, new_path)
        if self._save_score:
                save_score_dic(idx, spl_score_dic, frac_score_dic, hyb_score_dic, new_path)
        return tp_spl, fp_spl, spl_score_dic, tp_frac, fp_frac, frac_score_dic, tp_hyb, fp_hyb, hyb_score_dic