import math
from typing import List, Any

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import seaborn
from mpl_toolkits.mplot3d import Axes3D
import xlsxwriter
import pandas as pd
from sklearn import metrics

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x = pd.read_excel('N2-2(AE-3f).xlsx', header=None)
x = np.asarray(x)  # Convert to 2-D array

dir = 'x'
for c in range(x.__len__()):
    ax.scatter(x[c][0], x[c][1], x[c][2], c='k', zdir=dir)  # easy  y    diff x
plt.show()

flag = True
# mainRadius = 0.065
# radius = 0.065

mainRadius = 1.6
radius = 1.6

learning_rate = 0.9465
merge_num = 10

number_of_clusters = 2

oldCenters = []
xCenters = []
for i in range(len(x)):
    oldCenters.append(x[i])
    xCenters.append(x[i])

newCenters = []
# changeCenters = []

while (flag):
    print("r=", radius)
    for i in range(oldCenters.__len__()):
        in_bandwidth = []
        centroid = oldCenters[i]
        for c in oldCenters:
            if (math.fabs(c[0] - centroid[0]) + math.fabs(c[1] - centroid[1]) + math.fabs(
                    c[2] - centroid[2])) <= radius:
                in_bandwidth.append(c)
        x_c = 0
        y_c = 0
        z_c = 0
        for j in range(len(in_bandwidth)):
            x_c += in_bandwidth[j][0]
            y_c += in_bandwidth[j][1]
            z_c += in_bandwidth[j][2]
        x_c /= len(in_bandwidth)
        y_c /= len(in_bandwidth)
        z_c /= len(in_bandwidth)

        # changeCenters = [x_c,y_c]
        if [x_c, y_c, z_c] not in newCenters:
            newCenters.append([x_c, y_c, z_c])
        for k in range(xCenters.__len__()):
            if xCenters[k][0] == centroid[0] and xCenters[k][1] == centroid[1] and xCenters[k][2] == centroid[2]:
                xCenters[k] = [x_c, y_c, z_c]

    count = 0
    if newCenters.__len__() == oldCenters.__len__():
        for l in range(oldCenters.__len__()):
            if newCenters[l][0] == oldCenters[l][0] and newCenters[l][1] == oldCenters[l][1] and newCenters[l][2] == \
                    oldCenters[l][2]:
                count += 1
        if count == oldCenters.__len__():
            flag = False

    oldCenters = []
    for i in range(len(newCenters)):
        oldCenters.append(newCenters[i])

    newCenters.clear()

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111, projection='3d')
    for c in range(oldCenters.__len__()):
        ax2.scatter(oldCenters[c][0], oldCenters[c][1], oldCenters[c][2], c='r', zdir=dir)  # easy  y
    plt.show()

    radius *= learning_rate  # 0.95 0.7  0.99

fig3 = plt.figure()
ax3 = fig3.add_subplot(111, projection='3d')
rang = 1000 * ['g', 'r', 'b', 'y', 'c', 'm', 'k']
mainCentroid = np.zeros([len(oldCenters), 3])
numberOfSample = np.zeros(len(oldCenters))
labels = np.zeros(len(xCenters))
centerId = np.zeros(len(oldCenters))

print(len(oldCenters))
for i in range(len(oldCenters)):
    for j in range(len(xCenters)):
        if xCenters[j] == oldCenters[i]:
            ax3.scatter(x[j][0], x[j][1], x[j][2], c=rang[i], zdir=dir)  # easy  y
            mainCentroid[i][0] += x[j][0]
            mainCentroid[i][1] += x[j][1]
            mainCentroid[i][2] += x[j][2]
            numberOfSample[i] += 1
            labels[j] = i
            centerId[i] = i
    mainCentroid[i][0] /= numberOfSample[i]
    mainCentroid[i][1] /= numberOfSample[i]
    mainCentroid[i][2] /= numberOfSample[i]
    # ax3.text(mainCentroid[i][0], mainCentroid[i][1], mainCentroid[i][2],"Point 1", c='k', zdir='y') #, marker='*'
plt.show()

#  ----------------------------------merge----------------------------------
a = 0

for i in range(len(oldCenters)):
    if numberOfSample[i] < merge_num:  # 100
        min = 1000
        mergeBy = 0
        for j in range(len(oldCenters)):
            if numberOfSample[j] > merge_num:  # 100
                distance = math.fabs(mainCentroid[i][0] - mainCentroid[j][0]) + math.fabs(
                    mainCentroid[i][1] - mainCentroid[j][1]) + math.fabs(
                    mainCentroid[i][2] - mainCentroid[j][2])
                if distance < min:
                    min = distance
                    mergeBy = j
        centerId[i] = mergeBy
        for k in range(len(labels)):
            if labels[k] == i:
                labels[k] = mergeBy

fig4 = plt.figure()
ax4 = fig4.add_subplot(111, projection='3d')

remainCluster = []
c = 0
for i in range(len(labels)):
    ax4.scatter(x[i][0], x[i][1], x[i][2], c=rang[int(labels[i])], zdir=dir)  # easy  y
    if labels[i] not in remainCluster:
        remainCluster.append(labels[i])
        c += 1
plt.show()
print(c)
a = 0
#  --------------------------------accuracy------------------------------
a = 0
# ____________________________for easy diff data-------------------------
a = 0
# numberOfClass = 5
# myDic = [{}, {}, {}, {}, {}]  # find maximum label that assign to each real cluster
# for i in range(len(labels)):
#     j = i % numberOfClass
#     key = labels[i]
#     if key not in myDic[j]:
#         (myDic[j])[key] = 1
#     else:
#         (myDic[j])[key] += 1
#
# selectedLabels = []
# index = 6  # chon index 6 nadarim
# for i in range(numberOfClass):
#     key_list = list(myDic[i].keys())
#     val_list = list(myDic[i].values())
#
#     maxLabel = max(myDic[i].values())
#     key = key_list[val_list.index(maxLabel)]
#
#     if key in selectedLabels:  # bara peyda kardan id i ke 2 bar terar shode yani 2 ta kelase asli ba ham edgham shode
#         index = i
#     selectedLabels.append(key)
a = 0
# ---------------------------accuracy for real data---------------------
a = 0
true_label = pd.read_excel('E:\\D\\arshad\\terme4\\prj\\autoencoder\\w_real\\real_data\\N2-2-label.xlsx', header=None)
true_label = np.asarray(true_label)  # Convert to 2-D array

# -------- bara D o E ------------
# true_label = np.zeros([1, len(labels)])
# for i in range(len(true_label[0])):
#     true_label[0][i] = (i % 5) + 1
# --------------------------------


listFoundLabels = [[], [], [], [], []]
for i in range(len(true_label[0])):
    index = true_label[0][i] - 1
    listFoundLabels[int(index)].append(labels[i])

selectedLabels = []
countArray = []
for i in range(number_of_clusters):
    L = max(set(listFoundLabels[i]), key=listFoundLabels[i].count)  # find most frequent number in array
    count = listFoundLabels[i].count(L)  # find the number of most frequent number in array
    if L not in selectedLabels:
        selectedLabels.append(L)
        countArray.append(count)
    else:  # 2 ta khoshe edgham shodan va label yeksan daran.
        preL = selectedLabels.index(L)
        preCount = countArray[preL]
        if preCount > count:  # khoshe ghabli ke sabt shode bozorg tare
            listFoundLabels[i] = list(filter(lambda a: a != L, listFoundLabels[i]))  # remove all label which is L
            if len(listFoundLabels[i]) != 0:
                L = max(set(listFoundLabels[i]), key=listFoundLabels[
                    i].count)  # hala dovomin elementi ke bishtarin tekrar roo dare peyda mikonim va append
                count = listFoundLabels[i].count(L)
                selectedLabels.append(L)
                countArray.append(count)
            else:  # dar halati ke kole ye khoshe ba yeki dg edgham shode bashe va label e dg i behesh nakhorde bashe
                selectedLabels.append(100)
                countArray.append(0)
        else:  # khoshe jadide bozorgtare va khoshi ghabl bayad label esh avaz she
            selectedLabels.append(L)
            countArray.append(count)
            listFoundLabels[preL] = list(filter(lambda a: a != L, listFoundLabels[preL]))
            if len(listFoundLabels[preL]) != 0:
                newL = max(set(listFoundLabels[preL]), key=listFoundLabels[preL].count)
                newCount = listFoundLabels[preL].count(L)
                selectedLabels[preL] = newL
                countArray[preL] = newCount
            else:
                selectedLabels.append(100)
                countArray.append(0)

correct = 0
for i in range(len(true_label[0])):
    if labels[i] == selectedLabels[int(true_label[0][i] - 1)]:
        correct += 1
print("accuracy", correct / len(true_label[0]))
print("our: ", metrics.silhouette_score(x, labels, metric='euclidean'))

a = 0
# ---------------confusion matrix and accuracy for easy and diff data-------------
a = 1
# assignLabel = {}
# for i in range(len(selectedLabels)):
#     assignLabel[i] = selectedLabels[i]
#
# # selectedLabels = [2,1,4]
# confusionMatrix = np.zeros([numberOfClass, numberOfClass])
# correct = 0
# for i in range(len(labels)):
#     j = i % numberOfClass
#     if labels[
#         i] in selectedLabels:  # agar in label e jozve label haye aslie (az in kelasaye alaki ke index 10 o 11 ina daran nis
#         matrix_j = 0
#         matrix_i = 0
#         for key, value in assignLabel.items():  # for when a label is more than matrix label
#             if value == labels[i]:
#                 matrix_j = key
#             if value == selectedLabels[j]:
#                 matrix_i = key
#         if j == index:  # when two class merged
#             confusionMatrix[int(j)][int(matrix_j)] += 1
#         else:
#             confusionMatrix[int(matrix_i)][int(matrix_j)] += 1
#             if labels[i] == selectedLabels[j]:  # selectedLabels[j] == true label
#                 correct += 1
#
# print("TP", correct, "FP", (len(labels) - correct), "accuracy= ", correct / len(labels))

# seaborn.heatmap(confusionMatrix, annot=True, annot_kws={"size": 10}, fmt=".0f",
#                 cmap="YlGnBu")  # fmt adadado be float neshon bede o bedone ashar
# plt.show()

a = 1
# ----------------------------------test & online classification----------------------------------------
a = 1
# predicted_label = np.zeros(len(test_label))
# final_centroid = []
#
# first_time = 1
# for i in range(len(test)):
#     min_distance = 1000
#     label = 0
#     for j in range(len(remainCluster)):
#         dis = math.fabs(test[i][0] - oldCenters[int(remainCluster[j])][0]) + math.fabs(
#             test[i][1] - oldCenters[int(remainCluster[j])][1]) + math.fabs(
#             test[i][2] - oldCenters[int(remainCluster[j])][2])
#         if dis < min_distance:
#             min_distance = dis
#             predicted_label[i] = int(remainCluster[j])
#         if first_time == 1:
#             final_centroid.append(oldCenters[int(remainCluster[j])])
#     first_time = 0
#
# correct = 0
# for i in range(len(test_label)):
#     if selectedLabels[test_label[i]] == predicted_label[i]:
#         correct += 1
#
# print("test accuracy = ", correct / len(test))
#
# # write in file
# workbook = xlsxwriter.Workbook('cluster centers/5N/15-61E.xlsx')
# worksheetW = workbook.add_worksheet()
#
# row = 0
# col = 0
# for i in final_centroid:
#     for j in range(3):
#         worksheetW.write(row, col + j, i[col + j])
#     row += 1
#
# workbook.close()

# ______________similarity matrix_________
# df = pd.DataFrame(x, columns=["x1","x2","x3"])
# df['labels'] = labels
# df = df.sort_values(by=['labels'])
# df = df.drop(columns=['labels'])
# # similarity = metrics.pairwise.cosine_similarity(df, df, dense_output=True)
# from scipy import spatial
# similarity = spatial.distance.cosine(x, x)
#
# fig, ax = plt.subplots()
# cax = ax.matshow(similarity, interpolation='nearest')
# # ax.grid(True)
# plt.title('Similarity matrix')
# fig.colorbar(cax, ticks=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, .75, .8, .85, .90, .95, 1])
# plt.show()