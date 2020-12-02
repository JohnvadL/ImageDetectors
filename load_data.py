import math
import os
import cv2
import numpy as np
import matplotlib.image as img
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from numba import jit, njit, prange

def get_ck_code(subject_num, session_num, data):
    for index, row in data.iterrows():
        if int(row['Subject #']) == subject_num and int(row['Session #']) == session_num:
            return row['FACS Code']

def load_all_data(root_data_dir='./data/face_data'):
    if not os.path.isdir(root_data_dir):
        raise OSError('Data directory {} does not exist. Try to extract data from zip file first'.format(data_dir))

    label_map = {
        1:0,
        2:1,
        4: 2,
        5: 3,
        6: 4,
        7: 5,
        9: 6,
        10: 7,
        12: 8,
        14: 9,
        15: 10,
        17: 11,
        20: 12,
        23: 13,
        25: 14,
        26: 15,
        28: 16,
        43: 17,
    }
    # load ck+ data
    print("Loading ck+ data")
    ck_plus_data = []
    ck_plus_labels = []

    data_dir_path = root_data_dir + '/CK+/cohn-kanade-images'
    label_dir_path = root_data_dir + '/CK+/FACS_labels/FACS'

    if not os.path.isdir(data_dir_path) or not os.path.isdir(label_dir_path):
        raise OSError('Data directory for CK+ does not exist. Try and extract CK+ data from zip first')
    label_dir = os.listdir(label_dir_path)
    data_dir = os.listdir(data_dir_path)
    label_dir.sort()
    data_dir.sort()

    for x in range(len(data_dir)):
        subject_dir = data_dir[x]
        if os.path.isdir(os.path.join(data_dir_path, subject_dir)):
            # go into subject folder
            subject_path = os.path.join(data_dir_path, subject_dir)
            subject_label_path = os.path.join(label_dir_path, subject_dir)
            subject = int(subject_dir[1:])

            session_dir_paths = os.listdir(subject_label_path)
            for session_dir in session_dir_paths :
                # go into session folder
                session_data_path = os.path.join(subject_path, session_dir)
                session_label_path = os.path.join(subject_label_path, session_dir)
                if os.path.isdir(session_data_path):
                    try:
                        session_num = int(session_dir)
                        session_img = img.imread(os.path.join(session_data_path,os.listdir(session_data_path)[-1]))

                        label_file = open(os.path.join(session_label_path,os.listdir(session_label_path)[0]), 'r')
                        label_vector = np.zeros(18, dtype=int)

                        # extract necessary aus
                        for line in label_file:
                            l = line
                            au = int(float(l.split()[0]))
                            if au in label_map:
                                label_vector[label_map[au]] = 1
                        label_file.close()
                        ck_plus_labels.append([subject,session_num, label_vector])
                        ck_plus_data.append(session_img)
                    except:
                        continue
    print("Done loading ck plus data now loading ck data")

    # load ck data
    ck_dir = root_data_dir+'/CK/cohn-kanade'
    ck_codes = root_data_dir+'/CK/Cohn-Kanade Database FACS codes_updated based on 2002 manual_revised.xls'
    ck_data = []
    ck_labels = []

    ck_file = pd.read_excel(ck_codes, skiprows=[0,1,2])
    ck_frame = pd.DataFrame(ck_file)

    for subject_dir in os.listdir(ck_dir):
        # iterate subjects
        if os.path.isdir(os.path.join(ck_dir, subject_dir)):
            subject_path = os.path.join(ck_dir, subject_dir)
            subject = int(subject_dir[1:])

            for session_dir in os.listdir(subject_path):
                # iterate sessions
                session_path = os.path.join(subject_path, session_dir)
                if os.path.isdir(session_path):
                    session_num = int(session_dir)
                    try:
                        session_img = img.imread(os.path.join(session_path,os.listdir(session_path)[-1]))

                        # extract necessary aus
                        label = str(get_ck_code(subject, session_num, ck_frame))
                        if label == 'None':
                            continue
                        label_vector = np.zeros(18, dtype=int)
                        label_items = label.split('+')
                        for word in label_items:
                            au = int(''.join(c for c in word if c.isdigit()))
                            if au in label_map:
                                label_vector[label_map[au]] = 1
                        ck_labels.append([subject,session_num, label_vector])
                        ck_data.append(session_img)
                    except:
                        continue

    all_data = ck_plus_data + ck_data
    all_labels = ck_plus_labels + ck_labels

    # code to save to excel file
    # import xlwt
    # from xlsxwriter import Workbook
    # wb = Workbook('./FACS_labels.xsl')
    # sheet1 = wb.add_worksheet('Sheet 1')
    #
    #
    # for x in range(len(all_labels)):
    #     label = all_labels[x]
    #     r = [label[0], label[1]]
    #     r.extend(label[2])
    #     sheet1.write_row(x,0,r)
    #
    # wb.close()

    train_data, other_data,train_labels, other_labels = train_test_split(all_data, all_labels, test_size=0.3)
    test_data, valid_data, test_labels, valid_labels = train_test_split(other_data, other_labels, test_size=0.5)

    return train_data, train_labels, test_data, valid_data, test_labels, valid_labels




if __name__ == '__main__':
    load_all_data()