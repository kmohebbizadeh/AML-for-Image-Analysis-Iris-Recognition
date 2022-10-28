import os
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

folder = 'CASIA Iris Image Database (version 1.0)'
train_df = pd.DataFrame(columns=['id', 'eye1', 'eye2', 'eye3'])
test_df = pd.DataFrame(columns=['id', 'eye1', 'eye2', 'eye3', 'eye4'])

for person in os.listdir(folder):
    train_info = {'id': person}
    test_info = {'id': person}
    for file in os.listdir(folder + '/' + person):
        if file == '1':
            for picture in os.listdir(folder + '/' + person + '/' + '1'):
                if picture[-4:] != '.bmp':
                    continue
                path = folder + '/' + person + '/' + '1' + '/' + picture
                train_info['eye'+picture[-5]] = path
        if file == '2':
            for picture in os.listdir(folder + '/' + person + '/' + '2'):
                if picture[-4:] != '.bmp':
                    continue
                path = folder + '/' + person + '/' + '2' + '/' + picture
                test_info['eye'+picture[-5]] = path

    train_df = train_df.append(train_info, ignore_index=True)
    test_df = test_df.append(test_info, ignore_index=True)


train_df = train_df.sort_values(by=['id'])
test_df = test_df.sort_values(by=['id'])

