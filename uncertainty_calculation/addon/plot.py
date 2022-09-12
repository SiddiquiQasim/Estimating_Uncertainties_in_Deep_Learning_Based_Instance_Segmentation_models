import numpy as np
from warnings import warn
from stardist.utils import _normalize_grid

def random_label_cmap(n=2**16, h = (0,1), l = (.4,1), s =(.2,.8)):
    import matplotlib
    import colorsys
    h,l,s = np.random.uniform(*h,n), np.random.uniform(*l,n), np.random.uniform(*s,n)
    cols = np.stack([colorsys.hls_to_rgb(_h,_l,_s) for _h,_l,_s in zip(h,l,s)],axis=0)
    cols[0] = 0
    return matplotlib.colors.ListedColormap(cols)


def _plot_polygon_var(x,y,score,color):

    a,b = list(x),list(y)
    a += a[:1]
    b += b[:1]
    return a, b


def _plot_polygon(x,y,score,color):
    import matplotlib.pyplot as plt
    a,b = list(x),list(y)
    a += a[:1]
    b += b[:1]
    plt.plot(a,b,'--', alpha=1, linewidth=score, zorder=1, color=color)


def draw_polygons(coord, score, poly_idx, grid=(1,1), cmap=None, show_dist=False, median=False):
    """poly_idx is a N x 2 array with row-col coordinate indices"""
    grid = _normalize_grid(grid,2)
    return _draw_polygons(median, polygons=coord[poly_idx[:,0],poly_idx[:,1]],
                         points=poly_idx*np.array(grid),
                         scores=score[poly_idx[:,0],poly_idx[:,1]],
                         cmap=cmap, show_dist=show_dist, )


def _draw_polygons(median, polygons, points=None, scores=None, grid=None, cmap=None, show_dist=False):
    """
        polygons is a list/array of x,y coordinate lists/arrays
        points is a list/array of x,y coordinates
        scores is a list/array of scalar values between 0 and 1
    """
    # TODO: better name for this function?
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    if grid is not None:
        warn("parameter 'grid' has no effect anymore, please remove")
    if points is None:
        points = [None]*len(polygons)
    if scores is None:
        scores = np.ones(len(polygons))
    if cmap is None:
        cmap = random_label_cmap(len(polygons)+1)

    assert len(polygons) == len(scores)
    assert len(cmap.colors[1:]) >= len(polygons)
    assert not show_dist or all(p is not None for p in points)
    if not median:
        m_x = []
        m_y = []

    for point,poly,score,c in zip(points,polygons,scores,cmap.colors[1:]):
        if point is not None:
            if median:
                plt.plot(point[1], point[0], '.', markersize=8*score, color='red')
            else:
                plt.plot(point[1], point[0], '.', markersize=8*score, color='green')

        if show_dist:
            dist_lines = np.empty((poly.shape[-1],2,2))
            dist_lines[:,0,0] = poly[1]
            dist_lines[:,0,1] = poly[0]
            dist_lines[:,1,0] = point[1]
            dist_lines[:,1,1] = point[0]
            plt.gca().add_collection(LineCollection(dist_lines, linestyle='-', colors='red', linewidths=1))
        if median:
            _plot_polygon(poly[1], poly[0], 1.5, color='red')
                
        else:
            x, y = _plot_polygon_var(poly[1], poly[0], 3*score, color='green')
            m_x.append(x)
            m_y.append(y)

    if not median:
        return m_x, m_y

    
def plot_img_label(grt, img, lbl,grt_title='ground', img_title="image", lbl_title="label", **kwargs):
    import matplotlib.pyplot as plt
    fig, (gt,ai,al) = plt.subplots(1,3, figsize=(12,5), gridspec_kw=dict(width_ratios=(1,1,1)))
    gt.imshow(grt, cmap='gray', clim=(0,1))
    im = ai.imshow(img, cmap='gray', clim=(0,1))
    ai.set_title(img_title)    
    al.imshow(lbl, cmap='gray', clim=(0,1))
    al.set_title(lbl_title)
    plt.tight_layout()
    


def calibration_plot(tp, fp, spl, frac, hyb):
    import scipy
    import matplotlib.pyplot as plt
    from uncertainty_calculation.addon.calibration_error import cal_accuracy, detection_distribution, linear_regression, calibration_errors, expected_calibration_error

    unc, acc, _ = cal_accuracy(tp, fp)
    _, true_det, total_det = detection_distribution(tp, fp)
    x_r, y_r = linear_regression(tp, fp)
    ce, unc_mce, mce, acc_mce = calibration_errors(unc, acc)
    ece = expected_calibration_error(ce, total_det)
#     import pdb; pdb.set_trace()
    plt.figure(figsize=(6.5,6))
    plt.subplot(111)
    plt.plot([0,1], [0,1], color = 'green', linewidth=4)
    plt.plot(x_r, y_r, '--', color='aqua', linewidth=4)

    
    plt.bar( unc, acc, width = 0.1, color='mediumblue', align='center', edgecolor='black', alpha=1.0)
    plt.bar( unc, ce, bottom=acc, width = 0.1, linewidth=3, align='center', edgecolor='red', fc=(1, 1, 1, 0.05), hatch='//')
    plt.bar( unc_mce, mce, bottom=acc_mce, width = 0.1, linewidth=3, align='center', edgecolor='yellow', fc=(1, 1, 1, 0.05), hatch='xx')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylabel('Proportion Correct', fontsize=18)

    plt.text(0.03, 0.92, 'R : {:.3f}'.format(scipy.stats.pearsonr(unc, acc)[0]), fontsize=16, style='italic', bbox={'facecolor': 'aqua', 'alpha': 1.0, 'pad': 2})
    plt.text(0.03, 0.86, 'ECE : {:.3f}'.format(ece), fontsize=16, style='italic', bbox={'facecolor': 'red', 'alpha': 1.0, 'pad': 2})
    plt.text(0.03, 0.80, 'MCE : {:.3f}'.format(np.absolute(mce[0])), fontsize=16, style='italic', bbox={'facecolor': 'yellow', 'alpha': 1.0, 'pad': 2})
    if spl:
        plt.xlabel('Spatial Certainty', fontsize=18)
    elif hyb:
        plt.xlabel('Hybrid Certainty', fontsize=18)
    elif frac:
        plt.xlabel('Fraction Certainty', fontsize=18)

    plt.tight_layout()
    
    return