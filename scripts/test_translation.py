from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline


model_path = "./results"


tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)


src_lang = "tpi_Latn"
tgt_lang = "eng_Latn"


translator = pipeline(
    "translation",
    model=model,
    tokenizer=tokenizer,
    src_lang=src_lang,
    tgt_lang=tgt_lang
)


krio_input = "Di titi de wonshi"


translation = translator(krio_input)
print("Translation:", translation[0]["translation_text"])
