import os

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

def get_ck_code(subject_num, session_num, data):
    for index, row in data.iterrows():
        if int(row['Subject #']) == subject_num and int(row['Session #']) == session_num:
            return row['FACS Code']

def load_all_data(root_data_dir='../data/face_data'):
    if not os.path.isdir(root_data_dir):
        raise OSError('Data directory {} does not exist. Try to extract data from zip file first'.format(root_data_dir))

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
                        neutral_img = os.path.join(session_data_path,os.listdir(session_data_path)[0])
                        if neutral_img.find('DS') != -1:
                            neutral_img = os.path.join(session_data_path, os.listdir(session_data_path)[1])
                        session_img = os.path.join(session_data_path,os.listdir(session_data_path)[-1])

                        label_file = open(os.path.join(session_label_path,os.listdir(session_label_path)[0]), 'r')
                        label_vector = np.zeros(18, dtype=int)
                        neutral_vector = np.zeros(18, dtype=int)

                        # extract necessary aus
                        for line in label_file:
                            l = line
                            au = int(float(l.split()[0]))
                            if au in label_map:
                                label_vector[label_map[au]] = 1
                        label_file.close()
                        ck_plus_labels.append([subject,session_num, label_vector])
                        ck_plus_labels.append([subject, session_num, neutral_vector])
                        ck_plus_data.append(session_img)
                        ck_plus_data.append(neutral_img)
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
                        session_img = os.path.join(session_path,os.listdir(session_path)[-1])
                        neutral_img = os.path.join(session_path, os.listdir(session_path)[0])
                        if neutral_img.find('DS') != -1:
                            neutral_img = os.path.join(session_data_path, os.listdir(session_data_path)[1])

                        # extract necessary aus
                        label = str(get_ck_code(subject, session_num, ck_frame))
                        if label == 'None':
                            continue
                        label_vector = np.zeros(18, dtype=int)
                        neutral_vector = np.zeros(18, dtype=int)
                        label_items = label.split('+')
                        for word in label_items:
                            au = int(''.join(c for c in word if c.isdigit()))
                            if au in label_map:
                                label_vector[label_map[au]] = 1
                        ck_labels.append([subject,session_num, label_vector])
                        ck_labels.append([subject, session_num, neutral_vector])
                        ck_data.append(session_img)
                        ck_data.append(neutral_img)
                    except:
                        continue

    # group the images based on subject for ck
    ck_subject_dic = {}
    for x in range(len(ck_data)):
        subject, session, vector, img = 'ck_'+str(ck_labels[x][0]), ck_labels[x][1], ck_labels[x][2], ck_data[x]
        ck_subject_dic.setdefault(subject, [])
        ck_subject_dic[subject].append([session, vector, img])

    # group the images based on subject for ck+
    for x in range(len(ck_plus_data)):
        subject, session, vector, img = 'ck_plus_' + str(ck_plus_labels[x][0]),\
                                        ck_plus_labels[x][1], ck_plus_labels[x][2], ck_plus_data[x]
        ck_subject_dic.setdefault(subject, [])
        ck_subject_dic[subject].append([session, vector, img])

    # split data into test, train and valid sets
    subject_arr = np.array(list(ck_subject_dic.keys()))
    np.random.shuffle(subject_arr)

    train_subjects, test_subjects = train_test_split(subject_arr, test_size=0.05)
    batch2, batch1 = train_test_split(train_subjects, test_size=1/3)
    batch2, batch3 = train_test_split(batch2, test_size=0.5)

    train_subjects = [[],[],[]]
    valid_subjects = [[],[],[]]
    train_subjects[0], valid_subjects[0] = train_test_split(batch1, test_size=1/3)
    train_subjects[1], valid_subjects[1] = train_test_split(batch2, test_size=1/3)
    train_subjects[2], valid_subjects[2] = train_test_split(batch3, test_size=1/3)

    # populate labels and images
    test_data, test_labels = [], []
    for subject in test_subjects:
        for item in ck_subject_dic[subject]:
            session, vector, img = item
            test_data.append(img)
            test_labels.append([subject, session, vector])


    train_data = [[],[],[]]
    train_labels = [[],[],[]]
    for batch in range(3):
        for subject in train_subjects[batch]:
            for item in ck_subject_dic[subject]:
                session, vector, img = item
                train_data[batch].append(img)
                train_labels[batch].append([subject, session, vector])

    valid_data = [[],[],[]]
    valid_labels = [[],[],[]]
    for batch in range(3):
        for subject in valid_subjects[batch]:
            for item in ck_subject_dic[subject]:
                session, vector, img = item
                valid_data[batch].append(img)
                valid_labels[batch].append([subject, session, vector])

    return train_data, train_labels, test_data, valid_data, test_labels, valid_labels

def write_labels_to_csv(path, data, img_paths):
    # code to save to excel file
    file = open(path, 'w')

    column_names = [
        'subject',
        'session',
        'path',
        '1',
        '2',
        '4',
        '5',
        '6',
        '7',
        '9',
        '10',
        '12',
        '14',
        '15',
        '17',
        '20',
        '23',
        '25',
        '26',
        '28',
        '43'
    ]
    column_str = ','.join(column_names)
    column_str = ','+column_str+'\n'
    file.write(column_str)


    for x in range(len(data)):
        label = data[x]
        r = ['0',str(label[0]), str(label[1]), img_paths[x].split('\\')[-1]]
        r.extend(label[2].tolist())
        r = [str(x) for x in r]
        file.write(','.join(r)+'\n')

    file.close()
    return

def separate_data(train_data, train_labels, test_data, valid_data, test_labels, valid_labels):
    try:
        for batch in range(3):
            os.mkdir('./train_data_{}'.format(batch))
            os.mkdir('./valid_data_{}'.format(batch))
        os.mkdir('./test_data')
    except:
        print("Could not make directories maybe they already exist or no permission")
        print("May not be issue")

    # copy images to directories
    from shutil import copyfile
    for batch in range(3):
        for img in train_data[batch]:
            img_name = img.split('\\')[-1]
            copyfile(img, './train_data_{}/'.format(batch)+img_name)

    for img in test_data:
        img_name = img.split('\\')[-1]
        copyfile(img, './test_data/'+img_name)

    for batch in range(3):
        for img in valid_data[batch]:
            img_name = img.split('\\')[-1]
            copyfile(img, './valid_data_{}/'.format(batch)+img_name)

    # save the csv files
    for batch in range(3):
        write_labels_to_csv('./train_data_{}/labels.csv'.format(batch), train_labels[batch], train_data[batch])
        write_labels_to_csv('./valid_data_{}/labels.csv'.format(batch),valid_labels[batch], valid_data[batch])
    write_labels_to_csv('./test_data/labels.csv', test_labels, test_data)

class ImgDataset(Dataset):

    def __init__(self, data_path=''):
        """
        Args:
          - label_csv: Path to the csv file with action unit labels.
          - train: training set if True, otherwise validation set
          - intensity (bool): labels are intensities (between 0 and 5) rather
                              than presence (either 0 or 1).
          - transform: transform applied to an image input
        """
        label_path = data_path+'/labels.csv'
        self.root_dir = data_path

        self.au_frame = pd.read_csv(label_path,engine='python')
        self.label_cols = [
        '1',
        '2',
        '4',
        '5',
        '6',
        '7',
        '9',
        '10',
        '12',
        '14',
        '15',
        '17',
        '20',
        '23',
        '25',
        '26',
        '28',
        '43'
    ]

    def __len__(self):
        return len(self.au_frame)

    def __getitem__(self, idx):
        # Get image at idx
        image_id = self.au_frame.iloc[idx]['path']
        image_path = self.root_dir + '/' + image_id
        image = cv2.imread(image_path)

        # Get AU labels
        aus = self.au_frame.iloc[idx][self.label_cols]
        aus = np.array(aus, dtype=float)

        sample = {'image': image, 'labels': aus}

        return sample

if __name__ == '__main__':
    # Is used if data has not been sorted yet
    train_data, train_labels, test_data, valid_data, test_labels, valid_labels = load_all_data()
    separate_data(train_data, train_labels, test_data, valid_data, test_labels, valid_labels)
    # a = ImgDataset('./train_data_1')
    # print(a.__getitem__(0))
