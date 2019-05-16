import pandas as pd
import cv2 

def read_data():
    df = pd.read_csv('driving_log.csv', header=None, names=['Center', 'Left', 'Right', 'Steering', 'Throttle', 'Break', 'Speed'])
    df['Center'] = 'data/IMG/' + df['Center'].str.split('/').str[6]
    df['Name'] = df['Center'].str.split('_').str[1:]
    df['Name'] = df['Name'].apply('_'.join)
    # df['name'] = '_'.join(df['Center image'].str.split('_').str[1:])
    df['Left'] = 'data/IMG/' + df['Left'].str.split('/').str[6]
    df['Right'] = 'data/IMG/' + df['Right'].str.split('/').str[6]
    df = df[['Center', 'Left', 'Right', 'Name', 'Steering', 'Throttle', 'Break', 'Speed']]
    return df

df = read_data()
df.to_csv('driving_log_1.csv')
print(df[0:5])
img = cv2.imread('IMG/center_2019_05_12_08_56_25_090.jpg')

print(img.shape)
