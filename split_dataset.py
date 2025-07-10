import pandas as pd
from sklearn.model_selection import train_test_split
from ASVSpoofDataset import CreateCSVASVSpoof, create_balanced_dataset

if __name__ != '__main__':
    #In_The_Wild preparation
    data = pd.read_csv("C:\\Users\dmc\PycharmProjects\CASA-FVAB\processed_audio\chunkedDf.csv")
    data.sort_values(by='speaker', ascending=True, inplace=True)
    speaker = data['speaker'].unique()
    train_speaker, test_speaker = train_test_split(speaker, test_size=0.2, random_state=42)

    #ASVspoof_2021 preparation
    """
    Asvdf = pd.read_csv("C:\\Users\dmc\PycharmProjects\CASA-FVAB\cross_validation\ASVSpoofData.csv")
    folders = ['B:\\4835108\ASVspoof2021_DF_eval_part00\ASVspoof2021_DF_eval\\flac',
               'B:\\4835108\ASVspoof2021_DF_eval_part01\ASVspoof2021_DF_eval\\fflac',
               'B:\\4835108\ASVspoof2021_DF_eval_part02\ASVspoof2021_DF_eval\\flac',
               'B:\\4835108\ASVspoof2021_DF_eval_part03\ASVspoof2021_DF_eval\\flac']
    
    balanced_df = create_balanced_dataset(Asvdf, "./cross_validation/AsvspoofBalanced.csv",
                                          100000, folders)
    """
    balanced_asv = pd.read_csv("C:\\Users\dmc\PycharmProjects\CASA-FVAB\cross_validation\AsvspoofBalanced.csv")
    asv_balanced = pd.concat([balanced_asv[balanced_asv['label'] == 'bona-fide']
                                 , balanced_asv[balanced_asv['label'] == 'spoof'][:16842]])

