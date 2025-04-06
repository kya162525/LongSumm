import json
import os

path = "./abstractive_summaries/by_clusters/"


id_summary_map = {}
folders = os.listdir(path)
for folder in folders:
    folder_path = os.path.join(path, folder)
    if os.path.isdir(folder_path):
        files = os.listdir(folder_path)
        for file in files:
            if file.endswith(".json"):
                with open(os.path.join(folder_path, file), "r") as f:
                    data = json.load(f)

                id = data.get("id", "")
                summary = "\n".join(data.get("summary", ""))
                if id in id_summary_map:
                    id_summary_map[id] += "\n" + summary
                else:
                    id_summary_map[id] = summary
                print(f"Summary for {id} added.")

# Save the concatenated summaries to a new file
json.dump(id_summary_map, open("./id_summary_map.json", "w"), indent=4)
print(f"Concatenated summaries saved to id_summary_map.json")
