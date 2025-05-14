# pombo-labs-krio-english-translation-model

This project is a custom-built Natural Language Processing (NLP) pipeline that allows users to record audio in Krio (Creole), transcribe it, and then translate it to English using a fine-tuned translation model. The entire workflow was developed and tested using PyTorch, Hugging Face Transformers, and BLEU score evaluation for translation accuracy.

## Project Features

- Record Krio voice directly in the terminal (30 seconds max)
- Convert voice to text
- Translate Krio text to English using a fine-tuned model
- Evaluate model performance using BLEU score
- Trained with over **1,000+ Krio-English sentence pairs**

```bash
# Step 1: Install requirements
pip install -r requirements.txt

# Step 2: Record Krio audio
python3 scripts/record_audio.py

# Step 3: Transcribe audio to text
python3 scripts/transcribe_audio.py

# Step 4: Fine-tune model
python3 scripts/fine_tune_translation.py

# Step 5: Evaluate with BLEU
python3 scripts/bleu.py
```

## Project Structure & Workflow

### 1. Voice Recording

I created a script to record audio from the user via terminal:

````bash
python3 scripts/record_audio.py


- Records up to 30 seconds
- Saves the file as `output.wav`

### 2.  Transcribe Audio to Text

Use a speech-to-text model (e.g., `whisper`) to transcribe the recorded Krio audio into text.

run: whisper data/audio/my_test_audio.wav --model small --language en --task transcribe --output_dir data/transcripts/
to Transcribe


### 3.  Build Krio-English Dataset

- Created a `.jsonl` file with over **1,000+ validated sentence pairs**.
- Format:
```json
{"krio": "A de go na skul.", "en": "I am going to school."}
````

### 4. Fine-Tune Translation Model

Used Hugging Face Transformers to fine-tune `facebook/nllb-200-distilled-600M` on the Krio-English dataset.

```bash
python3 scripts/fine_tune_translation.py
```

- Trained using `Trainer`
- Model checkpoints saved in `./results/checkpoint-8535`

### 5. Evaluate Translation Accuracy (BLEU Score)

To test translation quality:

```bash
python3 scripts/bleu.py
```

- Evaluates predictions against reference translations
- Outputs BLEU score (mine scored 100.00)

## Project Folder Structure

pombo_nlp_project/
│
├── scripts/
│ ├── record_audio.py # For voice input
│ ├── fine_tune_translation.py # For model training
│ ├── bleu.py # For evaluation
│
├── krio_en_pairs_fixed.jsonl # Dataset
├── results/ # Fine-tuned model checkpoints
│ └── checkpoint-8500/
│ └── checkpoint-8535/

## Technologies Used

- Python 3.13
- Hugging Face Transformers
- PyTorch
- SacreBLEU
- Datasets library
- Sounddevice, Scipy (for audio input)

## Why This Project Matters

Krio is a low-resource language spoken by millions in Sierra Leone but underrepresented in AI. This project is a small but powerful step in building **inclusive language tools**, starting from my voice... to the world.

## Credits

- Created with passion and resilience by Alimatu Maliaka Jalloh
