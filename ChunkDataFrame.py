import os
import pandas as pd
import librosa
import soundfile as sf

# Configurazioni
INPUT_DIR = "release_in_the_wild"  # Cartella contenente i file originali
OUTPUT_DIR = "processed_audio"  # Cartella di output per i chunk
TARGET_DURATION = 5  # Durata target dei chunk in secondi

# Carica il dataframe con le durate
df = pd.read_csv("meta.csv")
df["duration"] = df["file"].apply(lambda x: librosa.get_duration(path=os.path.join(INPUT_DIR, x)))


# Funzione per processare gli speaker
def process_speakers(data):
    new_df = pd.DataFrame(columns=df.columns[:-1])
    for speaker in data["speaker"].unique():
        speaker_data = data[data["speaker"] == speaker]

        # Calcola mediana minima
        spoof_median = speaker_data[speaker_data["label"] == "spoof"]["duration"].median()
        bona_fide_median = speaker_data[speaker_data["label"] == "bona-fide"]["duration"].median()
        valid_medians = [m for m in [spoof_median, bona_fide_median] if not pd.isna(m)]
        #print(speaker, spoof_median, bona_fide_median, valid_medians)
        if not valid_medians:
            continue

        median_min = min(valid_medians)
        cap = int(median_min // TARGET_DURATION) if median_min > TARGET_DURATION else None

        # Processa ogni file dello speaker
        for _, row in speaker_data.iterrows():
            process_file(row, cap, speaker, new_df)
    new_df.to_csv(os.path.join(OUTPUT_DIR, "chunkedDf.csv"), index=False)


# Funzione per ritagliare i file
def process_file(row, cap, speaker, dataFrame):
    file_path = os.path.join(INPUT_DIR, row["file"])
    duration = row["duration"]

    try:
        y, sr = librosa.load(file_path, sr=None)
    except Exception as e:
        print(f"Errore nel caricare {file_path}: {e}")
        return

    # Calcola numero di chunk
    if cap:
        num_chunks = min(int(duration // TARGET_DURATION), cap)
    else:
        num_chunks = 1 if duration >= TARGET_DURATION else 0

    # Crea i chunk
    for i in range(num_chunks):
        start = i * TARGET_DURATION
        end = start + TARGET_DURATION
        chunk = y[int(start * sr):int(end * sr)]

        # Salva il chunk
        output_path = build_output_path(row, i, speaker, dataFrame)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        sf.write(output_path, chunk, sr)


# Funzione per generare i path di output
def build_output_path(original_row, chunk_num, speaker, dataFrame):
    base_name = os.path.splitext(os.path.basename(original_row['file']))[0]
    new_name = f"{base_name}_chunk{chunk_num}.wav"
    dataFrame.loc[-1] = [new_name, speaker, original_row['label']]
    dataFrame.index+=1
    dataFrame = dataFrame.sort_index(ascending=True)
    return os.path.join(OUTPUT_DIR, speaker, new_name)


# Esecuzione
if __name__ == "__main__":
    process_speakers(df)
    print("Processamento completato!")


