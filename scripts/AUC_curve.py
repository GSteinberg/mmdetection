import matplotlib.pyplot as plt
from collections import OrderedDict
import numpy as np
import os
import csv

## Call from mmdetection/scripts ##

threshold = []
prec = []
recall = []
f1 = []

for rep in os.scandir('../faster_rcnn_r101_fpn_1x_coco_results/num13/demo_results'): 
    # checking correct file extension
    if not rep.name.endswith('.csv'):
        continue

    # log threshold values
    th_val = rep.name.split('_')[-1][:-4]
    if len(th_val) == 2:
        th_val = float(th_val) / 100
    else:
        th_val = float(th_val) / 10

    threshold.append(th_val)

    # open csv
    with open(rep.path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        count = 0

        for row in csv_reader:
            if row[0] == 'prec':
                prec.append(row[1:])
            elif row[0] == 'recall':
                recall.append(row[1:])
            elif row[0] == 'f1':
                f1.append(row[1:])

# prec, recall, f1
data = OrderedDict()
for idx in range(len(threshold)):
    data[threshold[idx]] = [prec[idx], recall[idx], f1[idx]]

sorted_data = OrderedDict(sorted(data.items()))

prec_pfm = [float(el[0][0]) for el in sorted_data.values()]
prec_ksf = [float(el[0][1]) for el in sorted_data.values()]
prec_ave = [float(el[0][2]) for el in sorted_data.values()]
recall_pfm = [float(el[1][0]) for el in sorted_data.values()]
recall_ksf = [float(el[1][1]) for el in sorted_data.values()]
recall_ave = [float(el[1][2]) for el in sorted_data.values()]
f1_pfm = [float(el[2][0]) for el in sorted_data.values()]
f1_ksf = [float(el[2][1]) for el in sorted_data.values()]
f1_ave = [float(el[2][2]) for el in sorted_data.values()]
threshold = [float(el) for el in sorted_data.keys()]

# PLOT
fig, (prec, reca, f1) = plt.subplots(1, 3, figsize=(20,10))

prec.plot(threshold, prec_ave, color='blue', label='precision ave')
prec.plot(threshold, prec_pfm, color='red', label='precision pfm')
prec.plot(threshold, prec_ksf, color='green', label='precision ksf')

reca.plot(threshold, recall_ave, color='blue', label='recall ave')
reca.plot(threshold, recall_ksf, color='red', label='recall ksf')
reca.plot(threshold, recall_pfm, color='green', label='recall pfm')

f1.plot(threshold, f1_ave, color='blue', label='f1 ave')
f1.plot(threshold, f1_pfm, color='red', label='f1 pfm')
f1.plot(threshold, f1_ksf, color='green', label='f1 ksf')

ticks = [np.round(i, 2) for i in np.arange(0, 1.01, 0.05)]
for ax in [prec, reca, f1]:
    ax.set_yticks(ticks)
    ax.set_xlabel('score thresholds')

prec.set_ylabel('precision')
reca.set_ylabel('recall')
f1.set_ylabel('f1')

plt.legend()

plt.savefig('AUC_curve.png')
