
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve


def ManhattanDetector(data, sample, reps, users, t, t_selected):

    train, test, impostor = split_data(data, sample, users)
    mean_vector = []
    impostor_user = []
    genuine_user = []
    test_size = 400 - sample
    if reps == 0:
        reps = test_size

    for i in range(0, train.shape[0], sample):
        mean_vector.append(train.iloc[i:i + sample].mean())

    for i in range(0, impostor.shape[0], int((impostor.shape[0]) / len(users))):
        impostor_user.append(impostor.iloc[i:i + int((impostor.shape[0]) / len(users))])

    for i in range(0, test.shape[0], int((test.shape[0]) / len(users))):
        genuine_user.append(test.iloc[i:i + int((test.shape[0]) / len(users))])

    impostor_score = evaluateScore(mean_vector, impostor_user,users)
    genuine_score = evaluateScore(mean_vector, genuine_user,users)

    impostor_scor = convertdf(impostor_score)
    genuine_scor = convertdf(genuine_score)

    fpr = false_postive_rate(genuine_scor,t)
    ipr = impostor_pass_rate(impostor_scor,t)

    fpr_l = []
    ipr_l = []

    for i in t_selected:
        fpr_l.append(false_postive_rate(genuine_scor,i))
        ipr_l.append(impostor_pass_rate(impostor_scor,i))

    eet, equal_error = evaluateer(genuine_scor,impostor_scor)

    ipr_at_0 = impostor_pass_rate(impostor_scor,genuine_scor.max())
    fpr_at_0=false_postive_rate(genuine_scor,impostor_scor.min()*0.99)

    return fpr, ipr, fpr_l, ipr_l, ipr_at_0, fpr_at_0, impostor_scor, genuine_scor, eet, equal_error


def split_data(data,sample,users):
    train = pd.DataFrame()
    test = pd.DataFrame()
    impostor = pd.DataFrame()

    for user in users:
        temp = data.loc[data.subject == user]
        train = train.append(temp[:sample])
        test = test.append(temp[sample:])

    for user in users:
        temp = test.loc[test.subject != user, 'H.period':'H.Return']
        impostor = impostor.append(temp)

    return train.iloc[:, 3:], test.iloc[:, 3:], impostor


def convertdf(data):
    new_dataframe = pd.DataFrame()
    for i in range(len(data)):
        new_dataframe = new_dataframe.append(data[i].to_frame(name='Score'))

    return new_dataframe



def evaluateScore(template, score_list,users):
    score = []
    total = []
    for i in range(len(users)):
        score.append(abs(template[i] - score_list[i]))
        total.append(score[i].sum(axis=1))

    return total


def countcheck (data,t):
    predicted_labels = data.iloc[:] <= t
    predicted_label_list = predicted_labels['Score'].values.tolist()
    count_impostor = predicted_label_list.count(False)
    count_genuine=predicted_label_list.count(True)
    return count_impostor,count_genuine


def false_postive_rate(data,t):
    impostor_count,genuine_count=countcheck(data,t)
    fpr = impostor_count/len(data)

    return fpr


def impostor_pass_rate(data,t):
    impostor_count, genuine_count=countcheck(data,t)
    ipr = genuine_count/len(data)

    return ipr


def evaluateer(genuine,impostor):
    labels = [0] * len(genuine) + [1] * len(impostor)
    score = list(np.array(genuine['Score'])) + list(np.array(impostor['Score']))
    fpr, tpr, threshold = roc_curve(labels, score)
    ipr = 1-tpr
    eer_threshold = threshold[np.nanargmin(np.absolute(ipr - fpr))]
    eer = fpr[np.nanargmin(np.absolute((ipr - fpr)))]

    return eer, eer_threshold
