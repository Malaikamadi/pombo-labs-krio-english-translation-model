import json

input_path = "krio_en_pairs_fixed.jsonl"
output_path = "krio_en_pairs_cleaned.jsonl"

with open(input_path, "r", encoding="utf-8") as infile, open(output_path, "w", encoding="utf-8") as outfile:
    for i, line in enumerate(infile, 1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if "krio" in obj and "en" in obj and isinstance(obj["krio"], str) and isinstance(obj["en"], str):
                json.dump(obj, outfile)
                outfile.write("\n")
            else:
                print(f"[SKIP] Line {i} missing keys or invalid types")
        except json.JSONDecodeError as e:
            print(f"[ERROR] Line {i} has invalid JSON: {e}")

print("Cleaned file written to:", output_path)
