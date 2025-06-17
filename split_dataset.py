import pandas as pd
from sklearn.model_selection import train_test_split


if __name__ != '__main__':
    data = pd.read_csv("C:\\Users\dmc\PycharmProjects\CASA-FVAB\processed_audio\chunkedDf.csv")
    data.sort_values(by='speaker', ascending=True, inplace=True)
    speaker = data['speaker'].unique()
    train_speaker, test_speaker = train_test_split(speaker, test_size=0.2, random_state=42)
