"""Further sequentialize the EHR as GPT input"""

import csv
import os
import numpy as np
import pandas as pd
import copy
import pickle
import random
from datetime import datetime, timedelta


random.seed(2)


def difference_time_A_B_seconds(timeA, timeB, format_pattern="%m/%d/%Y %H:%M"):

    gap_seconds = (
        datetime.strptime(timeA, format_pattern)
        - datetime.strptime(timeB, format_pattern)
    ).total_seconds()

    return gap_seconds


def A_plus_hrs(timeA="05/12/2020 14:34", hr=2, format_pattern="%m/%d/%Y %H:%M"):

    r = (datetime.strptime(timeA, format_pattern) + timedelta(hours=hr)).strftime(
        format_pattern
    )

    return str(r)


test_time = "01/01/2019 00:00"


tau = 4
period_num = 18


feature_number = 6  # used to compute sparsity and missingness


data = pd.read_csv("all_vitals_period.csv")
data["for_training"] = data["Time"].apply(
    lambda x: int(
        difference_time_A_B_seconds(
            timeA=test_time, timeB=x, format_pattern="%m/%d/%Y %H:%M"
        )
        > 0
    )
)


print(data)
print(data._get_numeric_data().max())


training_impute = (
    data[data["for_training"] == 1]._get_numeric_data().median()
)  # only use the training data to compute median and impute. use _get_numeric_data() or the mean/median computation will be super slow due to the non-numerical columns!!!
print(training_impute)

print("mean value computed")


features_to_be_normalized_list = [
    "SBP",
    "DBP",
    "Pulse",
    "Temp",
    "SpO2",
    "Resp",
    "age",
]  # we will save and use this in the training process later
mean_std_dic = {}

training_mean = data[data["for_training"] == 1][features_to_be_normalized_list].mean()
training_std = data[data["for_training"] == 1][features_to_be_normalized_list].std()

for feature in features_to_be_normalized_list:
    mean_std_dic[feature] = {
        "mean": training_mean[feature],
        "std": training_std[feature],
    }

print(mean_std_dic)

pickle.dump(mean_std_dic, open("mean_std_dic.pkl", "wb"))

all_PID = sorted(data["PID"].unique())

data_new = []


sample_ind = 0  # we use this to uniquely identify a seq sample

for i, PID in enumerate(all_PID):
    print(i)

    data_PID_all_adm_index = data[data["PID"] == PID]

    adm_index_list = list(set(data_PID_all_adm_index["adm_index"]))

    for adm_index in adm_index_list:

        data_PID = copy.deepcopy(
            data_PID_all_adm_index[data_PID_all_adm_index["adm_index"] == adm_index]
        )
        data_PID = data_PID.reset_index(
            drop=True
        )  # reset the index, or they could be inconsistent due to the previous overlapping

        for ind in data_PID.index:
            # if ind+period_num-1 not in data_PID.index:
            if (
                ind + 2 - 1 not in data_PID.index
            ):  # in v5.5 we accept shorter seqs, as short as 2 periods
                continue

            if data_PID.loc[ind, "adm_start"] == 1:  # v5.5: seq len at most 12

                seq = copy.deepcopy(
                    data_PID.loc[ind : ind + period_num - 1]
                )  # pandas .loc index includes both bounds!
                if (
                    seq.isna().sum().sum() / (seq.shape[0] * feature_number) > 0.2
                ):  # use 0.1 for 4h gap to get more samples, also compute the missing ratio in this correct way
                    continue

                seq["sample_ind"] = sample_ind
                data_new.append(seq)

                sample_ind += 1


data_new = pd.concat(data_new)
data_new = data_new.fillna(training_impute)

print(data_new)

data_new.to_csv("all_vitals_seq.csv")
