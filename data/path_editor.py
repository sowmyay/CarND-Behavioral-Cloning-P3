import pandas as pd

def read_data():
    df = pd.read_csv('driving_log.csv', header=None, names=['Center image', 'Left image', 'Right image', 'Steering Angle', 'Throttle', 'Break', 'Speed'])
    df['Center image'] = df['Center image'].str.split('/').str[6]
    df['Name'] = df['Center image'].str.split('_').str[1:]
    df['Name'] = df['Name'].apply('_'.join)
    # df['name'] = '_'.join(df['Center image'].str.split('_').str[1:])
    df['Left image'] = df['Left image'].str.split('/').str[6]
    df['Right image'] = df['Right image'].str.split('/').str[6]
    df = df[['Center image', 'Left image', 'Right image', 'Name', 'Steering Angle', 'Throttle', 'Break', 'Speed']]
    return df

df = read_data()
print(df[0:5])
