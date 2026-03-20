import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import json
import os
import re
import argparse

def draw(result_json):
    json_data_all = [json.loads(line) for line in open(result_json, encoding='utf-8')]
    data = []
    for item in json_data_all:
        id = item["_id"]
        pattern = r"_len_(\d+)_"
        match = re.search(pattern, id)
        context_length = int(match.group(1)) if match else None
        pattern = r"depth_(\d+)"
        match = re.search(pattern, id)
        document_depth = eval(match.group(1))/100 if match else None
        score = item['score']
        # Appending to the list
        data.append({
            "Document Depth": document_depth,
            "Context Length": context_length,
            "Score": score
        })

    # Creating a DataFrame
    df = pd.DataFrame(data)

    pivot_table = pd.pivot_table(df, values='Score', index=['Document Depth', 'Context Length'], aggfunc='mean').reset_index() # This will aggregate
    pivot_table = pivot_table.pivot(index="Document Depth", columns="Context Length", values="Score") # This will turn into a proper pivot
    
    # Create a custom colormap. Go to https://coolors.co/ and pick cool colors
    cmap = LinearSegmentedColormap.from_list("custom_cmap", ["#F0496E", "#EBB839", "#0CD79F"])

    # Create the heatmap with better aesthetics
    plt.figure(figsize=(17.5, 8))  # Can adjust these dimensions as needed
    sns.heatmap(
        pivot_table,
        # annot=True,
        fmt="g",
        cmap=cmap,
        cbar_kws={'label': 'Score'},
        vmin=1,
        vmax=10,
    )

    # More aesthetics
    plt.title(f'Pressure Testing\nFact Retrieval Across Context Lengths ("Needle In A HayStack")')  # Adds a title
    plt.xlabel('Token Limit')  # X-axis label
    plt.ylabel('Depth Percent')  # Y-axis label
    plt.xticks(rotation=45)  # Rotates the x-axis labels to prevent overlap
    plt.yticks(rotation=0)  # Ensures the y-axis labels are horizontal
    plt.tight_layout()  # Fits everything neatly into the figure area
    # Show the plot
    plt.savefig(f"{os.path.splitext(result_json)[0]}.png")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--result_dir', type=str)
    args = parser.parse_args()
    result_dir = args.result_dir
    if not os.path.exists(result_dir):
        raise ValueError(f"Directory {result_dir} does not exist.")
    
    for result_json in os.listdir(result_dir):
        if result_json.endswith('.jsonl') and ('deepseek' in result_json.lower() or 'openai' in result_json.lower()):
            draw(os.path.join(result_dir, result_json))
            print(f"Draw {result_json} done")
    
