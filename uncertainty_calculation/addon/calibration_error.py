from sklearn.linear_model import LinearRegression
import numpy as np

def cal_accuracy(tp, fp):
    unc = []
    acc = []
    det = [] # if we have only detect false positive
    for i in tp:
        unc.append(float(i))
        try:
            acc.append(tp[i] / (tp[i] + fp[i]))
            det.append((tp[i] + fp[i]) / (tp[i] + fp[i]))
        except:
            unc.pop()
    return unc, acc, det


def detection_distribution(tp, fp):
    unc = []
    true_det = []
    total_det = []
    for i in tp:
        unc.append(float(i))
        try:
            true_det.append(tp[i])
            total_det.append(tp[i] + fp[i])
        except:
            unc.pop()
    true_det = [true_det[value] for value in range(len(true_det)) if total_det[value] != 0]
    total_det = [value for value in total_det if value != 0]
    return unc, true_det, total_det
    

def linear_regression(tp, fp):
    unc = []
    acc = []
    for i in tp:
        unc.append(float(i))
        if (tp[i] + fp[i]) != 0:
            acc.append(tp[i] / (tp[i] + fp[i]))
        else:
            unc.pop()
    reg = LinearRegression()
    reg.fit(np.array(unc).reshape(-1, 1), np.array(acc).reshape(-1, 1))
    return [0,1], reg.predict(np.array([0,1]).reshape(-1,1))


def calibration_errors(unc, acc):
    ce = np.array(unc) - np.array(acc) + (unc[1] - unc[0]) / 2
    mce = ce[np.where(np.absolute(ce) == np.max(np.absolute(ce)))]
    unc_mce = unc[np.where(ce == mce[0])[0][0]]
    acc_mce = acc[np.where(ce == mce[0])[0][0]]
    return ce, unc_mce, mce, acc_mce


def expected_calibration_error(ce, total_det):
    return np.sum((total_det / np.sum(total_det) ) * np.absolute(ce))