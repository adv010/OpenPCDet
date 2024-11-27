import numpy as np
import sys

ratio = float(sys.argv[1])
num = int(sys.argv[2])

with open("train.txt", "r") as f:
    lines = f.read().strip().split('\n')
    inds = np.random.choice(len(lines), int(len(lines) * ratio), replace=False)
    newlines = []
    for i in inds:
        newlines.append(f'{lines[i]} {i}')

with open("train_%.2f_%d.txt" % (ratio, num), "w") as fw:
    fw.write('\n'.join(newlines))

sample_id_list_lbl = [x.strip().split(' ')[0] for x in newlines]

# Load the full db_infos and filter out the instances that are not in the sample_id_list_lbl
with open("kitti_dbinfos_train.pkl", 'rb') as f:
    db_infos_full = pickle.load(f)
db_infos_lbl = defaultdict(list)
for sample_id in sample_id_list_lbl:
    for class_name in cfg.CLASS_NAMES:
        for instance in db_infos_full[class_name]:
            if instance['image_idx'] == sample_id:
                db_infos_lbl[class_name].append(instance)

with open(f"kitti_dbinfos_train_{ratio}_{num}", 'wb') as f:
    pickle.dump(db_infos_lbl, f)
