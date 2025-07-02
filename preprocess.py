import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm
import logging

from dataAudio import AudioConfig, AudioProcessor

# Assumiamo che AudioProcessor sia definito in un file chiamato 'audio_processor.py'

# --- Configurazione ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Directory di input (dove si trovano i .wav)
ROOT_AUDIO_DIR = Path('./processed_audio')
METADATA_CSV_PATH = ROOT_AUDIO_DIR / 'chunkedDf.csv'

# Directory di output (dove verranno salvati i .pt)
OUTPUT_FEATURES_DIR = Path('./processed_features')

def main():
    """
    Script per pre-calcolare le feature audio (Mel e Wav2Vec) e salvarle su disco.
    """
    if not METADATA_CSV_PATH.exists():
        logger.error(f"File metadata non trovato in: {METADATA_CSV_PATH}")
        return

    # 1. Crea la directory di output
    OUTPUT_FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Directory di output creata in: {OUTPUT_FEATURES_DIR}")

    # 2. Carica il DataFrame
    df = pd.read_csv(METADATA_CSV_PATH)
    logger.info(f"Trovati {len(df)} campioni nel file CSV.")

    # 3. Inizializza il processore audio
    # Assicurati che i parametri (sample_rate, n_mels, etc.) siano gli stessi
    # che userai durante l'addestramento.
    audio_conf=AudioConfig()
    processor = AudioProcessor(audio_conf, wav2vec_model_name="facebook/wav2vec2-large-xlsr-53")

    # 4. Itera, elabora e salva
    num_processed = 0
    num_skipped = 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Elaborazione file audio"):
        speaker_id = row['speaker']
        file_name = row['file']
        
        audio_path = ROOT_AUDIO_DIR / speaker_id / file_name
        
        if not audio_path.exists():
            logger.warning(f"File non trovato: {audio_path}. Salto.")
            num_skipped += 1
            continue

        # Crea la sottodirectory di output se non esiste
        output_speaker_dir = OUTPUT_FEATURES_DIR / speaker_id
        output_speaker_dir.mkdir(exist_ok=True)

        # Definisci il percorso del file di output .pt
        feature_file_path = output_speaker_dir / f"{Path(file_name).stem}.pt"

        # (Opzionale) Salta se il file è già stato elaborato
        if feature_file_path.exists():
            continue

        try:
            # Carica l'audio
            waveform = processor.load_audio(audio_path)
            if waveform is None or waveform.numel() == 0:
                logger.warning(f"Caricamento fallito o file vuoto per {audio_path}. Salto.")
                num_skipped += 1
                continue

            # Estrai le feature
            features_dict = processor.extract_features(waveform)

            # Salva il dizionario di feature
            torch.save(features_dict, feature_file_path)
            num_processed += 1

        except Exception as e:
            logger.error(f"Errore durante l'elaborazione di {audio_path}: {e}")
            num_skipped += 1
    
    logger.info("--- Pre-processing completato! ---")
    logger.info(f"File elaborati con successo: {num_processed}")
    logger.info(f"File saltati: {num_skipped}")

if __name__ == '__main__':
    main()