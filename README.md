

Krio-English Speech Translation Pipeline

Project Overview

This project focuses on building a Krio-to-English translation system that can transcribe spoken Krio and convert it accurately into English text. It involves recording Krio speech, transcribing the audio using Whisper, translating the transcripts with a fine-tuned translation model, and evaluating the results using BLEU score. The final goal is to develop a functional machine translation pipeline that can understand Creole speech and deliver accurate English translations.

Step 1 - Choosing a programming language . I choose Python
Why Python?
Python was chosen for this project due to its robust ecosystem in machine learning and natural language processing. Libraries like HuggingFace Transformers, datasets, and sacrebleu provide mature APIs that make development more efficient. Python’s popularity in the AI research community and support for rapid prototyping make it ideal for NLP projects like this.

Setup Instructions

Ensure Python 3.3 is installed.

Clone this repository and create a virtual environment:

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

Data Source & Preparation

Voice samples were recorded in Krio using a custom Python script with the sounddevice library. Each clip is 30 seconds long and saved as WAV files. The Whisper model is used to transcribe these audio files into Krio text. Transcripts were reviewed and corrected manually to ensure accuracy.

1. Voice Recording
I created a script to record audio from the user via terminal:

python3 scripts/record_audio.py


- Records up to 30 seconds
- Saves the file as `output.wav`



Transcription Process

Model Used: OpenAI’s Whisper (installed via openai-whisper)

Why Whisper? Whisper offers robust transcription even on low-resource languages like Krio. It's end-to-end and requires minimal preprocessing.

Alternatives considered: Vosk and Mozilla DeepSpeech, but Whisper was chosen due to better multilingual performance.
### 2.  Transcribe Audio to Text

Use a speech-to-text model (e.g., `whisper`) to transcribe the recorded Krio audio into text.

run: whisper data/audio/my_test_audio.wav --model small --language en --task transcribe --output_dir data/transcripts/
to Transcribe


How to transcribe:

whisper myaudio.wav --language kr --model small

Translation & Pairing Process

Each Krio transcript was paired with its English equivalent to form sentence pairs. Some sentences were created manually to supplement the dataset. We ensured semantic equivalence between Krio and English lines. The final dataset is stored in krio_en_pairs.jsonl, containing over 1,000 validated examples.

NMT Fine-tuning & Evaluation

a. Base Model

facebook/nllb-200-distilled-600M was used for its lightweight architecture and multilingual support.

Why this model? Compared to larger NLLB models, this variant allows faster fine-tuning on local machines with acceptable accuracy.


Data Split

90% training

10% evaluation (manually split in the fine-tuning script)

c. Fine-tuning Script
### 3.  Build Krio-English Dataset

- Created a `.jsonl` file with over **1,000+ validated sentence pairs**.
- Format:
```json
{"krio": "A de go na skul.", "en": "I am going to school."}


The training was done using HuggingFace Trainer. To run the script:

python scripts/fine_tune_translation.py

Key Hyperparameters

Epochs: 3

Batch size: 4

Learning rate: 5e-5
Evaluation with BLEU

BLEU evaluation was done using sacrebleu:

python scripts/bleu.py

BLEU Score: 100.00 (on toy test set)

Note: This score reflects a small sample set and is not representative of full test accuracy.



Challenges & Assumptions

Limited resources: Krio is a low-resource language, making it hard to find datasets.

Overheating and time limits: Fine-tuning on local machines (MacBook Pro M4) caused long runtimes and thermal throttling.

File size: Full project exceeded 60GB due to large model checkpoints and audio files.

Manual cleaning: Transcripts required manual corrections post-Whisper to ensure accuracy.

Potential Improvements

Use Google Colab Pro or a cloud GPU to reduce training time.

Expand the dataset with community recordings.

Integrate UI for easier recording and translation.

Switch to NLLB-1.3B for higher accuracy if cloud compute is available.

Visualize translation accuracy with confusion matrices or attention maps.

Final Notes

This project shows a working prototype of a Krio-English speech translation pipeline using local tools and models. All components are offline and reproducible without needing cloud APIs or authentication tokens.

