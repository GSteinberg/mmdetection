import matplotlib.pyplot as plt
from collections import OrderedDict
import numpy as np
import os
import csv
import sys

## Call from mmdetection/scripts ##

threshold = []
prec = []
recall = []
f1 = []

INPUT_DIR = sys.argv[1]
TITLE = sys.argv[2]

for rep in os.scandir(INPUT_DIR):
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
fig, (pfm, ksf, ave) = plt.subplots(1, 3, figsize=(20,10))

pfm.plot(threshold, prec_pfm, color='blue', label='pfm precision')
pfm.plot(threshold, recall_pfm, color='red', label='pfm recall')
pfm.plot(threshold, f1_pfm, color='green', label='pfm f1')

ksf.plot(threshold, prec_ksf, color='blue', label='ksf precision')
ksf.plot(threshold, recall_ksf, color='red', label='ksf recall')
ksf.plot(threshold, f1_ksf, color='green', label='ksf f1')

ave.plot(threshold, prec_ave, color='blue', label='ave precision')
ave.plot(threshold, recall_ave, color='red', label='ave recall')
ave.plot(threshold, f1_ave, color='green', label='ave f1')

ticks = [np.round(i, 2) for i in np.arange(0, 1.01, 0.1)]
for ax in [pfm, ksf, ave]:
    ax.set_yticks(ticks)
    ax.set_xticks(ticks)
    ax.set_xlabel('Score Thresholds')
    ax.legend()

pfm.set_ylabel('PFM Metrics')
ksf.set_ylabel('KSF Metrics')
ave.set_ylabel('Average Metrics')

ksf.set_title(TITLE)

plt.savefig('AUC_curve.png')
