"""Extract the clinical variables from raw structured EHR in csv files"""

import csv
import os
import numpy as np
import pandas as pd
import copy
import pickle
import random
from datetime import datetime, timedelta
import re


random.seed(2)


raw_data_path = "/data2/brhao/anomaly_project/BMC_box_data/"


vitals_filename_list = [
    raw_data_path + "bp100_ts.csv",
    raw_data_path + "bp120_ts.csv",
    raw_data_path + "bp140_ts.csv",
    raw_data_path + "bp160_ts.csv",
    raw_data_path + "bp180_ts.csv",
    raw_data_path + "vitals100_ts.csv",
    raw_data_path + "vitals110_ts.csv",
    raw_data_path + "vitals120_ts.csv",
    raw_data_path + "vitals140_ts.csv",
    raw_data_path + "vitals150_ts.csv",
    raw_data_path + "vitals165_ts.csv",
    raw_data_path + "vitals180_ts.csv",
    raw_data_path + "vitals195_ts.csv",
]


adm_filename = raw_data_path + "drg_ts.csv"
icu_filename = raw_data_path + "icuts.csv"  # new icu data in v5.4
intu_filename = raw_data_path + "vent_ts.csv"
death_filename = raw_data_path + "deaths_ts.csv"

demo_filename = raw_data_path + "demo.csv"  # add demo in v5.2

pmh_filename = (
    raw_data_path + "problem_list_fixed.csv"
)  # add pmh in v5.7. we fixed the bugs in the raw csv file

covid_filename = (
    raw_data_path + "covid_labs_taken_time.csv"
)  # v7: spicemen collected time added and used


# ----------------------------- process demo


def extract_demo_info(demo_filename):

    demo_data = pd.read_csv(demo_filename)
    demo_data_dic = {}

    print(demo_data)

    for i in demo_data.index:
        PID = str(demo_data.loc[i, "ID"])
        brithday = demo_data.loc[i, "BIRTH_DT"]
        gender_female = int(demo_data.loc[i, "GENDER"] == "F")
        hispanic = int(
            demo_data.loc[i, "HISPANIC_IND_NM"] == "Yes - Hispanic or Latino"
        )
        race_black = int(
            demo_data.loc[i, "PRIMARY_RACE_NM"] == "Black / African American"
        )
        race_white = int(demo_data.loc[i, "PRIMARY_RACE_NM"] == "White")
        race_other = int(
            demo_data.loc[i, "PRIMARY_RACE_NM"]
            not in ["Black / African American", "White"]
        )
        language_eng = int(demo_data.loc[i, "LANG_NM"] == "English")

        if PID not in demo_data_dic:

            demo_data_dic[PID] = {
                "birthday": brithday + " 00:00",
                "gender_female": gender_female,
                "hispanic": hispanic,
                "race_black": race_black,
                "race_white": race_white,
                "race_other": race_other,
                "language_eng": language_eng,
            }

    return demo_data_dic


demo_data_dic = extract_demo_info(demo_filename=demo_filename)

print(demo_data_dic["100001"])
print(demo_data_dic["100002"])


# ----------------------------- process PMH


def difference_time_A_B_seconds(timeA, timeB, format_pattern="%m/%d/%Y %H:%M"):

    gap_seconds = (
        datetime.strptime(timeA, format_pattern)
        - datetime.strptime(timeB, format_pattern)
    ).total_seconds()

    return gap_seconds


icd10_dic = {
    "diabetes": ["E08", "E09", "E10", "E11", "E13"],
    "htn": ["I10", "I15", "I16"],
    "ckd": ["I12", "N18"],
    "chd": ["I25"],
    "vd_deficiency": ["E55"],
    "obesity": ["E66"],
    "exam_with_abnormal": ["Z00.01"],
    "medical_facilities": ["Z75"],
    "reflux": ["K21"],
    "anemia": ["D60", "D61", "D62", "D63", "D64"],
    "other_specified_health_status": ["Z78"],
    "other_specified_counseling": ["Z71.89"],
    "personal_risk_factors": ["Z91"],
}


initial_pmh_dic = {
    "diabetes": "2/22/2222 00:00",
    "htn": "2/22/2222 00:00",
    "ckd": "2/22/2222 00:00",
    "chd": "2/22/2222 00:00",
    "vd_deficiency": "2/22/2222 00:00",
    "obesity": "2/22/2222 00:00",
    "exam_with_abnormal": "2/22/2222 00:00",
    "medical_facilities": "2/22/2222 00:00",
    "reflux": "2/22/2222 00:00",
    "anemia": "2/22/2222 00:00",
    "other_specified_health_status": "2/22/2222 00:00",
    "other_specified_counseling": "2/22/2222 00:00",
    "personal_risk_factors": "2/22/2222 00:00",
}


def extract_pmh_info(pmh_filename):

    pmh_data = [item for item in csv.reader(open(pmh_filename, "r", encoding="utf-8"))]
    pmh_data_dic = {}

    for line in pmh_data[1:]:

        line = line[0].split(";")

        if len(line) != 6:

            print("invalid line")
            continue

        PID = line[0]
        icd10 = line[3][1:-1]
        time = line[4]

        year = time.split("/")[2]
        month = time.split("/")[0]

        day = time.split("/")[1][len(month) :]

        time_reformatted = month + "/" + day + "/" + year + " 00:00"

        if PID not in pmh_data_dic:

            pmh_data_dic[PID] = copy.deepcopy(initial_pmh_dic)

        icd10_list = icd10.split()

        for code in icd10_list:

            code_major = code.split(".")[
                0
            ]  # only keep the primary part of icd10 code, like I50.30 becomes I50

            for disease in pmh_data_dic[PID]:
                if (
                    code_major in icd10_dic[disease] or code in icd10_dic[disease]
                ) and difference_time_A_B_seconds(
                    timeA=time_reformatted,
                    timeB=pmh_data_dic[PID][disease],
                    format_pattern="%m/%d/%Y %H:%M",
                ) < 0:  # v6.1: some icd10 extracted by Yang need to be more specific

                    pmh_data_dic[PID][disease] = time_reformatted

    return pmh_data_dic


pmh_data_dic = extract_pmh_info(pmh_filename=pmh_filename)


for PID in demo_data_dic:
    if (
        PID not in pmh_data_dic
    ):  # make the pmh have complete PID keys, to avoid bugs later

        pmh_data_dic[PID] = copy.deepcopy(initial_pmh_dic)


print(pmh_data_dic["100001"])


# ----------------------------- process covid labs


def extract_covid_info(covid_filename):

    covid_data = [
        item for item in csv.reader(open(covid_filename, "r", encoding="utf-8"))
    ]
    covid_data_dic = {}

    for line in covid_data[1:]:
        PID = line[0]
        order_time = line[2]
        result_time = line[3]
        specimen_taken_time = line[7]

        if line[5].lower() == "positive":

            if specimen_taken_time != "":
                time = specimen_taken_time
            else:
                if (
                    difference_time_A_B_seconds(
                        timeA=result_time,
                        timeB=order_time,
                        format_pattern="%m/%d/%Y %H:%M",
                    )
                    <= 30 * 24 * 3600
                ):
                    time = order_time
                else:
                    print(
                        "specimen_taken_time N/A, and result time "
                        + result_time
                        + " too far away from order time "
                        + order_time
                    )
                    continue

            if PID not in covid_data_dic:
                covid_data_dic[PID] = []

            covid_data_dic[PID].append(time)

    return covid_data_dic


covid_data_dic = extract_covid_info(covid_filename=covid_filename)

print(covid_data_dic["100003"])


feature_dic_empty = {
    "SBP": [],
    "DBP": [],
    "Pulse": [],
    "Temp": [],
    "SpO2": [],
    "Resp": [],
    "BMI (Calculated)": [],
    "Height": [],
    "Weight": [],
    "adm_records": [],
    "icu_records": [],
    "intu_records": [],
    "death_records": [],
}


feature_range_dic = {
    "SBP": {"upper": 500.0, "lower": 0.0},
    "DBP": {"upper": 500.0, "lower": 0.0},
    "Pulse": {"upper": 1000.0, "lower": 0.0},
    "Temp": {
        "upper": 150.0,
        "lower": 32.0,
    },  # 113898 has a temp=0.3 in record, which could raise errors! so we set lower=32F=0C rather than 0
    "SpO2": {"upper": 100.0, "lower": 0.0},
    "Resp": {"upper": 500.0, "lower": 0.0},
    "BMI (Calculated)": {"upper": 1000.0, "lower": 0.0},
    "Height": {"upper": 1000.0, "lower": 0.0},
    "Weight": {"upper": 50000.0, "lower": 0.0},
}


data_dic = {}


# ------------------------------ collect vitals features


for filename in vitals_filename_list:
    print("now processing " + filename)

    lines = [item for item in csv.reader(open(filename, "r", encoding="utf-8"))]

    for i, line in enumerate(lines[1:]):
        line = line[0].split(";")

        PID = line[0]
        time = line[1]
        feature_name = line[2][1:-1]
        feature_value = line[3][1:-1]

        if PID not in data_dic:
            data_dic[PID] = copy.deepcopy(feature_dic_empty)

        if feature_name == "BP":

            if feature_value.find("/") == -1:
                print(PID + " has invalid BP records")
                continue

            for j, bp_name in enumerate(["SBP", "DBP"]):

                if (
                    float(feature_value.split("/")[j])
                    >= feature_range_dic[bp_name]["lower"]
                    and float(feature_value.split("/")[j])
                    <= feature_range_dic[bp_name]["upper"]
                ):
                    data_dic[PID][bp_name].append(
                        {"time": time, "value": float(feature_value.split("/")[j])}
                    )

                else:
                    print(
                        str(float(feature_value.split("/")[j]))
                        + " is not a valid "
                        + bp_name
                    )

        else:

            if (
                float(feature_value) >= feature_range_dic[feature_name]["lower"]
                and float(feature_value) <= feature_range_dic[feature_name]["upper"]
            ):
                data_dic[PID][feature_name].append(
                    {"time": time, "value": float(feature_value)}
                )

            else:
                print(str(float(feature_value)) + " is not a valid " + feature_name)


# ------------------------------ collect adm information


print("now processing " + adm_filename)

lines = [item for item in csv.reader(open(adm_filename, "r", encoding="utf-8"))]

for i, line in enumerate(lines[1:]):

    line = line[0].split(";")

    PID = line[0]
    time_start = line[1]
    time_end = line[2]

    if line[4] != "" and line[5] != "":
        drg_code = int(float(line[4][1:-1]))  # 1:-1 exclude ""
        drg_type = line[5][1:-1]
    else:
        drg_code = -99
        drg_type = "None"

    if time_start == "" or time_end == "":
        continue

    if PID not in data_dic:
        data_dic[PID] = copy.deepcopy(feature_dic_empty)

    data_dic[PID]["adm_records"].append(
        {
            "time": time_start,
            "time_end": time_end,
            "value": 1,
            "drg_code": (drg_type, drg_code),
        }
    )  # in v5.6, also record the drg code for adm reasons


# ------------------------------ collect icu information


print("now processing " + icu_filename)

lines = [item for item in csv.reader(open(icu_filename, "r", encoding="utf-8"))]

for i, line in enumerate(lines[1:]):

    line = line[0].split(";")
    PID = line[0]

    time_start = line[2]  # new format in icuts.csv
    time_end = line[3]

    if time_start == "" or time_end == "":
        continue

    if PID not in data_dic:
        data_dic[PID] = copy.deepcopy(feature_dic_empty)

    data_dic[PID]["icu_records"].append(
        {"time": time_start, "time_end": time_end, "value": 1}
    )


# ------------------------------ collect intu information


print("now processing " + intu_filename)

lines = [item for item in csv.reader(open(intu_filename, "r", encoding="utf-8"))]

for i, line in enumerate(lines[1:]):

    times = re.findall(
        r"(\d{1,4}\/\d{1,4}\/\d{1,4}\ \d{2}\:\d{2})", line[0]
    )  # becasue there are many ";" in intu data line, we have to use re

    PID = line[0].split(";")[1]
    time_start = times[1]
    time_end = times[2]

    if time_start == "" or time_end == "":
        continue

    if PID not in data_dic:
        data_dic[PID] = copy.deepcopy(feature_dic_empty)

    data_dic[PID]["intu_records"].append(
        {"time": time_start, "time_end": time_end, "value": 1}
    )


# ------------------------------ collect death information


print("now processing " + death_filename)

lines = [item for item in csv.reader(open(death_filename, "r", encoding="utf-8"))]

for i, line in enumerate(lines[1:]):

    line = line[0].split(";")

    PID = line[0]
    time_start = line[1]
    time_end = line[1]  # use death time as both start and end to unify the format

    if time_start == "" or time_end == "":
        continue

    if PID not in data_dic:
        data_dic[PID] = copy.deepcopy(feature_dic_empty)

    data_dic[PID]["death_records"].append(
        {"time": time_start, "time_end": time_end, "value": 1}
    )


print(data_dic["100001"])
print(len(data_dic))


pickle.dump(data_dic, open("all_vitals.pkl", "wb"))
pickle.dump(demo_data_dic, open("all_demo.pkl", "wb"))
pickle.dump(pmh_data_dic, open("all_pmh.pkl", "wb"))
pickle.dump(covid_data_dic, open("all_covid.pkl", "wb"))


for PID in data_dic:
    if PID not in demo_data_dic:
        print(
            "PID with no demo information:", PID
        )  # seems like there is PID no missing demo, which is good
