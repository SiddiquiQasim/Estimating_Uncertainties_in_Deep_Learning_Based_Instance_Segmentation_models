import yaml

def tp_fp_save_all(idx, tp_spl, fp_spl, tp_frac, fp_frac, tp_hyb, fp_hyb, info, calibration_output):

    print('spl')
    with open(calibration_output + 'm_spl_tp.yml', 'w') as yaml_file:
        yaml.dump(tp_spl, yaml_file, default_flow_style=False)
    with open(calibration_output + 'm_spl_fp.yml', 'w') as yaml_file:
        yaml.dump(fp_spl, yaml_file, default_flow_style=False)
    print('frac')
    with open(calibration_output + 'frac_tp.yml', 'w') as yaml_file:
        yaml.dump(tp_frac, yaml_file, default_flow_style=False)
    with open(calibration_output + 'frac_fp.yml', 'w') as yaml_file:
        yaml.dump(fp_frac, yaml_file, default_flow_style=False)
    print('hyb')
    with open(calibration_output + 'hyb_tp.yml', 'w') as yaml_file:
        yaml.dump(tp_hyb, yaml_file, default_flow_style=False)
    with open(calibration_output + 'hyb_fp.yml', 'w') as yaml_file:
        yaml.dump(fp_hyb, yaml_file, default_flow_style=False)
    with open(calibration_output + 'all_info.yml', 'w') as yaml_file:
        yaml.dump(info, yaml_file, default_flow_style=False)
    return

def save_score_dic(idx, spl_score_dic, frac_score_dic, hyb_score_dic, calibration_output):

    print('spl')
    with open(calibration_output + 'spl_score_dic.yml', 'w') as yaml_file:
        yaml.dump(spl_score_dic, yaml_file, default_flow_style=False)
    print('frac')
    with open(calibration_output + 'frac_score_dic.yml', 'w') as yaml_file:
        yaml.dump(frac_score_dic, yaml_file, default_flow_style=False)
    print('hyb')
    with open(calibration_output + 'hyb_score_dic.yml', 'w') as yaml_file:
        yaml.dump(hyb_score_dic, yaml_file, default_flow_style=False)
    return