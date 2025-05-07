import json

input_file = "krio_en_pairs.jsonl"  
output_file = "cleaned_krio_en_pairs.jsonl"

with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
    for line_number, line in enumerate(infile, 1):
        line = line.strip()
        if not line:
            continue
        try:
            # Try to parse the line as JSON
            json_obj = json.loads(line)
            if "krio" in json_obj and "en" in json_obj:
                json.dump(json_obj, outfile)
                outfile.write("\n")
            else:
                print(f"[SKIPPED] Line {line_number} missing 'krio' or 'en'")
        except json.JSONDecodeError as e:
            print(f"[ERROR] Line {line_number} has invalid JSON: {e}")
