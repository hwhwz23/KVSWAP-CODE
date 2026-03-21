import json
import re
import os
import argparse

parser = argparse.ArgumentParser(description='Calculate scores from JSON files')
parser.add_argument('--path', type=str, required=True, help='Path to the directory containing JSON files')
args = parser.parse_args()

path = args.path

def extract_scores(text):
    # Define the keys to locate in the text
    keys = ["score_completeness", "score_reliability"]
    scores = []

    for key in keys:
        # Find the index where each key starts
        start_index = text.find(key)
        if start_index == -1:
            continue  # Skip if key is not found

        # Find the start of the number which is after the colon and space
        start_number_index = text.find(":", start_index) + 2
        end_number_index = text.find(",", start_number_index)  # Assuming the number ends before a comma

        try:
            # Extract and convert the number to float
            score = float(text[start_number_index:end_number_index])
            scores.append(score)
        except:
            print(f"Error extracting score for {key} in {text}")
            print("Skip this sample")
            return None

    return scores



accu=0
rele=0
total=0
file_list=os.listdir(path)


skip_count=0

for i in file_list:
    file_path=os.path.join(path,i)
    with open(file_path,"r") as f:
        data=json.load(f)

    # print(file_path)
    text=data[0]["explain"]
    # print(text)
    scores=extract_scores(text)
    # print("score",scores)
    if scores is None:
        skip_count += 1
        continue
    try:
        accu += scores[0]
        rele += scores[1]
    except:
        accu +=0
        rele+=0
  

if len(file_list) == 0:
    print("No files found in the directory.")
    exit(0)

accu = accu/ (len(file_list) - skip_count)
rele = rele/ (len(file_list) - skip_count)
total= (accu + rele ) 

print(f"Average Completeness Score: {accu:.2f}")
print(f"Average Reliability Score: {rele:.2f}")
print(f"Total Score: {total:.2f}")
print(f"Total number of files: {len(file_list)}")