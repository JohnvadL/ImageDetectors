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

def load_all_data(root_data_dir='./data/face_data'):
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
                        session_img = os.path.join(session_data_path,os.listdir(session_data_path)[-1])

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
                        session_img = os.path.join(session_path,os.listdir(session_path)[-1])

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

    train_data, other_data,train_labels, other_labels = train_test_split(all_data, all_labels, test_size=0.3)
    test_data, valid_data, test_labels, valid_labels = train_test_split(other_data, other_labels, test_size=0.5)

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
         os.mkdir('./train_data')
         os.mkdir('./test_data')
         os.mkdir('./valid_data')
    except:
        print("Could not make directories maybe they already exist or no permission")
        print("May not be issue")

    # copy images to directories
    from shutil import copyfile
    for img in train_data:
        img_name = img.split('\\')[-1]
        copyfile(img, './train_data/'+img_name)

    for img in test_data:
        img_name = img.split('\\')[-1]
        copyfile(img, './test_data/'+img_name)

    for img in valid_data:
        img_name = img.split('\\')[-1]
        copyfile(img, './valid_data/'+img_name)

    # save the csv files
    write_labels_to_csv('./train_data/labels.csv', train_labels, train_data)
    write_labels_to_csv('./test_data/labels.csv',test_labels, test_data)
    write_labels_to_csv('./valid_data/labels.csv', valid_labels,valid_data)

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
    #train_data, train_labels, test_data, valid_data, test_labels, valid_labels = load_all_data()
    #separate_data(train_data, train_labels, test_data, valid_data, test_labels, valid_labels)
    a = ImgDataset('./train_data')
    print(a.__getitem__(0))
