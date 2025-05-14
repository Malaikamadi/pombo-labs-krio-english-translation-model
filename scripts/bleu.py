from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from datasets import load_dataset
from sacrebleu import corpus_bleu
import os


model_path = "./results/checkpoint-8535"


if not os.path.isdir(model_path):
    raise FileNotFoundError(f"Model directory not found: {model_path}")

# Load model and tokenizer from local directory
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path, local_files_only=True)

# Setup translation pipeline
translator = pipeline(
    "translation",
    model=model,
    tokenizer=tokenizer,
    src_lang="eng_Latn",  
    tgt_lang="eng_Latn",
    device=-1  
)


test_data = [
    {"krio": "Ar de waka go di shop.", "en": "I am walking to the shop."},
    {"krio": "Mi mama de cook.", "en": "My mother is cooking."},
    {"krio": "Di san don set.", "en": "The sun has set."}
]


sources = [item["krio"] for item in test_data]
references = [[item["en"]] for item in test_data]

# Translate
predictions = []
for text in sources:
    output = translator(text, max_length=128)[0]["translation_text"]
    predictions.append(output.strip())

# Evaluate BLEU score
bleu = corpus_bleu(predictions, references)
print(f"\nBLEU Score: {bleu.score:.2f}")
