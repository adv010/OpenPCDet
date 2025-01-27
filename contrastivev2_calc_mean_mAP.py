#!/usr/bin/python3
# python calc_mean_mAP.py --save_to_file --exp_names disabled_st_bs4_trial1_818cd7c disabled_st_bs4_trial2_818cd7c disabled_st_bs4_trial3_818cd7c

import argparse
import os
import pickle
import re
import glob
import numpy as np
import shutil


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--exp_names', required=True, nargs=argparse.REMAINDER,
                        help='--exp_names <test-name-1>, <test-name-2> ..')
    parser.add_argument('--thresh', type=str, default='0.5, 0.25, 0.25')
    parser.add_argument('--save_to_file', action='store_true', default=True, help='')
    parser.add_argument('--log_tb', action='store_true', default=False, help='')
    parser.add_argument('--result_tag', type=str, default=None, help='extra tag for this experiment')
    args = parser.parse_args()
    return args


def get_sorted_text_files(dirpath):
    a = [s for s in os.listdir(dirpath)
        if os.path.isfile(os.path.join(dirpath, s)) and s.endswith('.txt')]
    a.sort(key=lambda s: os.path.getmtime(os.path.join(dirpath, s)))
    return a

def find_eval_list_file(exp_name):
    primary_path = os.path.join("output/cfgs/kitti_models/pv_rcnn_ssl_60", exp_name, "eval/eval_with_student_model/eval_list_val.txt")
    secondary_path = os.path.join("output/kitti_models/pv_rcnn_ssl_60", exp_name, "eval/eval_with_student_model/eval_list_val.txt")
    tertiary_path = os.path.join("output/kitti_models/pv_rcnn_ssl_60", exp_name, "eval/eval_all_default/default/eval_list_val.txt")
    if os.path.isfile(primary_path):
        return primary_path, 1
    elif os.path.isfile(secondary_path):
        return secondary_path, 2
    elif os.path.isfile(tertiary_path): #using eval_all 
        log_path = os.path.join("output/kitti_models/pv_rcnn_ssl_60", exp_name, "eval/eval_all_default/default")
                # Find log files matching the pattern log_eval_*.txt
        log_files = glob.glob(os.path.join(log_path, "log_eval_*.txt"))
        
        if log_files:
            # Assuming we want to copy all matching log files
            destination_dir = os.path.join("output/kitti_models/pv_rcnn_ssl_60", exp_name)
            os.makedirs(destination_dir, exist_ok=True)  # Create destination directory if it doesn't exist
            for log_file in log_files:
                shutil.copy(log_file, destination_dir)  # Copy each log file to the destination directory
        return tertiary_path, 3
    else:
        return None, -1

def find_res_dir(exp_name):

    primary_path = os.path.join("output/cfgs/kitti_models/pv_rcnn_ssl_60", exp_name)
    secondary_path = os.path.join("output/kitti_models/pv_rcnn_ssl_60", exp_name)
    if os.path.isdir(primary_path):
        return primary_path
    elif os.path.isdir(secondary_path):
        return secondary_path
    else:
        return None

def calc_mean_mAP():
    """
    Takes n experiments and calculate mean of max class mAP
    """
    args = parse_config()
    #THRESH_ = [float(x) for x in args.thresh.split(',')]
    assert args.exp_names is not None
    exp_names = [str(x) for x in args.exp_names]

    metric = ["Car AP_R40@0.70, 0.70, 0.70", "Pedestrian AP_R40@0.50, 0.50, 0.50", "Cyclist AP_R40@0.50, 0.50, 0.50"]
    pattern = re.compile(r'({0})'.format('|'.join(metric)))
    max_results = []
    max_results2 = []
    eval_list = None

    print("\n#--------------------Calculate Mean mAP-----------------------#\n")
    print("\nDefined Metric")
    for m in metric:
        print(m)
    print("\nExperiment(s))")
    for e in exp_names:
        print(e)

    if args.save_to_file:
        res_text_file = os.path.join(os.getcwd(), "{}_results.txt".format(exp_names[0] if args.result_tag is None else exp_names[0] + args.result_tag))
        fw = open(res_text_file, 'w')
        fw.write("\n#--------------------Calculate Mean mAP-----------------------#\n")
        fw.write("\nDefined Metric\n")
        fw.write(str(metric))
        fw.write("\nExperiment(s)\n")
        fw.write(str(exp_names))
    all_eval_results = []
    for _exp in exp_names:
        curr_eval_list_file, exp_type = find_eval_list_file(_exp)
        if eval_list is None and curr_eval_list_file is not None:
            with open(curr_eval_list_file) as f_eval:
                eval_list = list(set(map(int, f_eval.readlines()))) # take only unique entries
                eval_list_selected = [epoch for epoch in eval_list if epoch in [50, 55, 60]]  # filter for specific epochs
                print("\nEvaluated Epochs")
                print(*[str(i) for i in eval_list], sep=",")
                if args.save_to_file:
                    fw.write("\nEvaluated Epochs")
                    fw.write(str(eval_list))

        curr_res_dir = find_res_dir(_exp)
        if curr_res_dir is None:
            print("No result directory found for {}".format(_exp))
            continue

        text_files = get_sorted_text_files(curr_res_dir)
        if len(text_files) == 0:
            print("No text file found containing results")
            continue

        eval_results = []
        eval_results2 = []

        if exp_type == 3: # eval_all
            for file_ in text_files: # traverse all file to find evaluation results
                selected_file = os.path.join(curr_res_dir, file_)
                print("\nScanning {} for evaluated results\n".format(selected_file)) # can be filtered based on date-time
                if args.save_to_file:
                    fw.write("\nScanning {} for evaluated results\n".format(selected_file))

                line_numbers = []
                linenum = 0

                with open(selected_file) as fp:
                    for line in fp:
                        linenum += 1
                        if pattern.search(line) is not None: # If a match is found
                            line_numbers.append(linenum + 3) # add following res-line-number into list
                        if linenum in line_numbers:
                            res_ = np.fromstring(line.strip().split("3d   AP:")[1], dtype=np.float64, sep=',')
                            eval_results.append(res_)

        else:
            for file_ in text_files: # traverse all file to find evaluation results
                selected_file = os.path.join(curr_res_dir, file_)
                with open(selected_file, 'r') as f:
                    content = f.read()
                
                # Split content at the start of teacher evaluation
                parts = content.split("**********************Start evaluation for teacher model")
                intro = content.split("**********************Start evaluation for student model")[0]
                # Create student log file
                student_file = selected_file.replace('.txt', '_student.txt')
                with open(student_file, 'w') as sf:
                    sf.write(parts[0])
                    sf.write("\n********************** Reading Student model")
                print("\nScanning {} for evaluated results\n".format(student_file)) # can be filtered based on date-time
                if args.save_to_file:
                    fw.write("\nScanning {} for evaluated results\n".format(student_file))
                teacher_file = selected_file.replace('.txt', '_teacher.txt')
                with open(teacher_file, 'w') as tf:
                    tf.write(intro)
                    tf.write(parts[1])
                    tf.write("\n**********************Reading Teacher model")
                print("\nScanning {} for evaluated results\n".format(teacher_file)) # can be filtered based on date-time
                if args.save_to_file:
                    fw.write("\nScanning {} for evaluated results\n".format(teacher_file))

                line_numbers = []
                linenum = 0
                line_numbers2 = []
                linenum2 = 0

                with open(student_file) as fp:
                    for line in fp:
                        linenum += 1
                        if pattern.search(line) is not None: # If a match is found
                            line_numbers.append(linenum + 3) # add following res-line-number into list
                        if linenum in line_numbers:
                            res_ = np.fromstring(line.strip().split("3d   AP:")[1], dtype=np.float64, sep=',')
                            eval_results.append(res_)

                with open(teacher_file) as tp:
                    for line in tp:
                        linenum2 += 1
                        if pattern.search(line) is not None: # If a match is found
                            line_numbers2.append(linenum2 + 3) # add following res-line-number into list
                        if linenum2 in line_numbers2:
                            res_ = np.fromstring(line.strip().split("3d   AP:")[1], dtype=np.float64, sep=',')
                            eval_results2.append(res_)

                if args.save_to_file:
                    fw.write("\n****OUTPUT SCANNING COMPLETE****")
                    print("\n****OUTPUT SCANNING COMPLETE****")
    
        #Write Student mAP of _exps onto file
        eval_results_full = eval_results
        eval_results_full = np.array(eval_results_full).reshape(len(eval_list), -1)
        eval_results = eval_results[-9:]  # consider only last 3 ckpts [50,55,60]
        eval_results = np.array(eval_results).reshape(len(eval_list_selected), -1)

        # all_eval_results.append(eval_results)
        print("\nStudent mAP(s)")
        print(*[str(np.round_(i, decimals=2)) for i in eval_results], sep="\n")

        if args.save_to_file:
            fw.write("\n Student mAP(s)")
            fw.write(str(np.round_(eval_results, decimals=2)))

        current_max = np.max(eval_results_full, axis=0)
        max_results.append(current_max)
        print("\nMax Student mAP")
        print(*[str(np.round_(i, decimals=2)) for i in current_max], sep=", ")
        if args.save_to_file:
            fw.write("\nMax Student mAP")
            fw.write(str(np.round_(current_max, decimals=2)))
        print("\n\n")        
        
        # Write Teacher mAP of _exps onto file
        eval_results_full2 = eval_results2
        eval_results_full2 = np.array(eval_results_full2).reshape(len(eval_list), -1)
        eval_results2 = eval_results2[-9:]
        eval_results2 = np.array(eval_results2).reshape(len(eval_list_selected), -1)
        # eval_results2.append(eval_results2)
        print("\nTeacher mAP(s)")
        print(*[str(np.round_(i, decimals=2)) for i in eval_results2], sep="\n")
        
        if args.save_to_file:
            fw.write("\nTeacher mAP(s)")
            fw.write(str(np.round_(eval_results2, decimals=2)))

        current_max2 = np.max(eval_results_full2, axis=0)
        max_results2.append(current_max2)
        print("\nMax Teacher mAP")
        print(*[str(np.round_(i, decimals=2)) for i in current_max2], sep=", ")
        if args.save_to_file:
            fw.write("\nMax Teacher mAP")
            fw.write(str(np.round_(current_max2, decimals=2)))
        print("\n\n")


    print("\n\n----------------Final Results----------------\n\n")
    max_results = np.array(max_results)
    print("Max Student mAP(s)\n")
    print(*[str(np.round_(i, decimals=2)) for i in max_results], sep="\n")

    if args.save_to_file:
        fw.write("\n\n----------------Final Results----------------\n\n")
        fw.write("Max Student mAP(s)\n")
        fw.write(str(np.round_(max_results, decimals=2)))

    mean_res = np.mean(eval_results, axis=0) #Modify to get mean of evaled checkpoints 
    print("\nMean Student mAP[50,55,60]")
    print(*[str(np.round_(i, decimals=2)) for i in mean_res], sep=", ")
    if args.save_to_file:
        fw.write("\nMean Student mAP[50,55,60]")
        fw.write(str(np.round_(mean_res, decimals=2)))

    stddev_res = np.std(eval_results, axis=0) #Modify to get mean of evaled checkpoints 
    print("\nStd.Dev of Student mAP [50,55,60]")
    print(*[str(np.round_(i, decimals=2)) for i in stddev_res], sep=", ")
    if args.save_to_file:
        fw.write("\n Std.Dev of Student mAP [50,55,60]")
        fw.write(str(np.round_(stddev_res, decimals=2)))


    #Write Teacher mAP onto file
    

    print("\n\n----------------Final Results----------------\n\n")
    max_results2 = np.array(max_results2)
    print("Max Teacher mAP(s)\n")
    print(*[str(np.round_(i, decimals=2)) for i in max_results2], sep="\n")

    if args.save_to_file:
        fw.write("\n\n----------------Final Results----------------\n\n")
        fw.write("Max Teacher mAP(s)\n")
        fw.write(str(np.round_(max_results2, decimals=2)))

    mean_res2 = np.mean(eval_results2, axis=0)
    print("\nMean Teacher mAP[50,55,60]")
    print(*[str(np.round_(i, decimals=2)) for i in mean_res2], sep=", ")
    if args.save_to_file:
        fw.write("\nMean Teacher mAP[50,55,60]")
        fw.write(str(np.round_(mean_res2, decimals=2)))


    stddev_res2 = np.std(eval_results2, axis=0) #Modify to get mean of evaled checkpoints 
    print("\nStd.Dev of Student mAP [50,55,60]")
    print(*[str(np.round_(i, decimals=2)) for i in stddev_res2], sep=", ")
    if args.save_to_file:
        fw.write("\n Std.Dev of Teacher mAP [50,55,60]")
        fw.write(str(np.round_(stddev_res2, decimals=2)))


    if args.log_tb:
        from tensorboardX import SummaryWriter

        re_trial = re.compile(r'trial(\d)')
        trials = re_trial.findall(" ".join(exp_names))
        trials = sorted(map(int, trials))
        trial_splits = re_trial.split(exp_names[0])
        new_trial = "trial{0}-{1}".format(str(trials[0]), str(trials[-1]))
        new_exp = "".join([trial_splits[0], new_trial, trial_splits[-1]])
        new_exp_dir = os.path.join("output/cfgs/kitti_models/pv_rcnn_ssl_60", new_exp, "eval", "eval_with_train",
                                   "tensorboard_val")
        all_eval_results = np.dstack(all_eval_results)
        mean_eval_results = np.mean(all_eval_results, -1)

        classes = ['Car_3d', 'Pedestrian_3d', 'Cyclist_3d']
        difficulties = ['easy_R40', 'moderate_R40', 'hard_R40']
        num_diffs = len(difficulties)
        num_classes = len(classes)
        class_wise_mean_eval_results = mean_eval_results.reshape((-1, num_diffs, num_classes), order='F')

        tb_log = SummaryWriter(log_dir=new_exp_dir)
        for i, cls in enumerate(classes):
            for j, diff in enumerate(difficulties):
                key = cls + "/" + diff
                for k, step in enumerate(eval_list):
                    val = class_wise_mean_eval_results[k, j, i]
                    tb_log.add_scalar(key, val, step)

if __name__ == "__main__":
    calc_mean_mAP()
