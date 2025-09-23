# Fake Audio Detection – Computer Vision Exam Project  

This repository contains the project developed for the **Computer Vision exam**, focused on applying **Machine Learning** techniques for **fake vs. real audio classification**.  
The goal was to explore multiple models and fusion strategies (late fusion and intermediate fusion) to improve detection performance.  

---

## Project Overview  

With the rapid growth of generative models, the ability to distinguish between **real** and **synthetically generated audio** has become increasingly important.  
In this project, we implemented and compared different approaches for audio classification, combining both traditional Machine Learning algorithms and deep learning architectures.  

We explored:  
- **Single-model classifiers** (baseline and advanced models).  
- **Fusion methods** to combine multiple models’ outputs:  
  - **Late Fusion**: combining final predictions.  
  - **Intermediate Fusion**: merging latent feature representations.  

---

## Models Implemented  

- **ResNet** – adapted for spectrogram-based classification.  
- **Wav2Vec2** – a pretrained transformer-based model for raw audio representation.  
- **Random Forest (RF)** – classical ML approach for baseline comparison.  
- **Support Vector Machine (SVM)** – traditional ML with audio features.  
- **CNN + RNN hybrid** – convolutional layers for feature extraction + recurrent layers for temporal modeling.  

---

## Tech Stack  

**Languages & Frameworks**  
- Python 3.x  
- PyTorch (deep learning)  
- scikit-learn (classical ML)  

**Audio Processing & Utilities**  
- Librosa  
- NumPy / Pandas  
- Matplotlib / Seaborn (visualization)  

 
