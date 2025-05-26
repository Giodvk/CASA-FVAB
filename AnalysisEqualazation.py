import pandas as pd
import librosa

# Carica il CSV
df = pd.read_csv('meta.csv')

# Aggiungi una colonna 'duration' con la durata dei file .wav
def get_duration(file_path):
    file_dir = "./release_in_the_wild/"
    return librosa.get_duration(path=file_dir + file_path)

df['duration'] = df['file'].apply(get_duration)

print(df.head())

results = {}

for speaker in df['speaker'].unique():
    speaker_data = df[df['speaker'] == speaker]

    # Separa clip fake e real
    spoof = speaker_data[speaker_data['label'] == 'spoof']
    bona_fide = speaker_data[speaker_data['label'] == 'bona-fide']

    # Calcola le mediane
    medians = []
    if not spoof.empty:
        medians.append(spoof['duration'].median())
    if not bona_fide.empty:
        medians.append(bona_fide['duration'].median())

    if not medians:
        continue  # Salta speaker senza clip

    median_min = min(medians)

    # Calcola il numero di sottosequenze per clip
    total_chunks = 0
    for _, row in speaker_data.iterrows():
        duration = row['duration']

        if median_min > 5:
            cap = int(median_min // 5)
            chunks = min(int(duration // 5), cap)
        else:
            chunks = 1 if duration >= 5 else 0

        total_chunks += chunks

    results[speaker] = total_chunks
print(results)