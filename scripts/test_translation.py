from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# Path to your fine-tuned model
model_path = "./results"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

# Define source and target languages
src_lang = "tpi_Latn"   # Set this to the right Krio-compatible code, e.g., "tpi_Latn"
tgt_lang = "eng_Latn"

# Create the translation pipeline
translator = pipeline(
    "translation",
    model=model,
    tokenizer=tokenizer,
    src_lang=src_lang,
    tgt_lang=tgt_lang
)

# Sample input
krio_input = "Di titi de wonshi"

# Generate translation
translation = translator(krio_input)
print("Translation:", translation[0]["translation_text"])
