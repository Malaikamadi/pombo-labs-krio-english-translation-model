from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from datasets import load_dataset
from sacrebleu import corpus_bleu

# Set path to local model checkpoint
model_path = "./results/checkpoint-best"

# Load model and tokenizer locally
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path, local_files_only=True)

# Set up pipeline for translation
translator = pipeline("translation", model=model, tokenizer=tokenizer, src_lang="eng_Latn", tgt_lang="eng_Latn", device=-1)

# Load your test set
test_data = [
    {"krio": "Ar de waka go di shop.", "en": "I am walking to the shop."},
    {"krio": "Mi mama de cook.", "en": "My mother is cooking."},
    {"krio": "Di sun don set.", "en": "The sun has set."}
]

# Extract source and reference sentences
sources = [item["krio"] for item in test_data]
references = [[item["en"]] for item in test_data]  # BLEU expects list of lists

# Run translations
predictions = []
for text in sources:
    output = translator(text, max_length=128)[0]["translation_text"]
    predictions.append(output.strip())

# Compute BLEU
bleu = corpus_bleu(predictions, references)
print(f"\nBLEU Score: {bleu.score:.2f}")
