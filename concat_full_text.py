import json
import os


path = "./papers/postprocessed/jsons/"
files = os.listdir(path)

# Create a directory to save the full texts
if not os.path.exists(path.replace("jsons", "full_texts")):
    os.makedirs(path.replace("jsons", "full_texts"))

for file in files:
    if file.endswith(".json"):
        with open(os.path.join(path, file), "r") as f:
            data = json.load(f)

        # Concatenate all text fields
        full_text = ""
        full_text += data.get("title", "") + "\n"
        full_text += data.get("abstractText", "") + "\n"

        for section in data.get("sections", []):
            full_text += section.get("heading", "") + "\n"
            full_text += section.get("text", "") + "\n"

        # Save the full text to a new file
        output_file = os.path.join(path.replace("jsons", "full_texts"), file.replace(".json", ".txt"))
        with open(output_file, "w") as f:
            f.write(full_text)
        print(f"Full text saved to {output_file}")
