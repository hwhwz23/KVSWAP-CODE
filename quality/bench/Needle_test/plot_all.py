import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import json
import re
from matplotlib.ticker import FuncFormatter

def draw(ax, result_json, show_cbar=False, cbar_ax=None):

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


    df = pd.DataFrame(data)

    pivot_table = pd.pivot_table(df, values='Score', index=['Document Depth', 'Context Length'], aggfunc='mean').reset_index() # This will aggregate
    pivot_table = pivot_table.pivot(index="Document Depth", columns="Context Length", values="Score") # This will turn into a proper pivot
    
    # Create a custom colormap. Go to https://coolors.co/ and pick cool colors
    # cmap = LinearSegmentedColormap.from_list("custom_cmap", ["#F0496E", "#EBB839", "#0CD79F"])
    
    cmap = plt.get_cmap("Greys_r")

    # sns.heatmap(
    #     pivot_table,
    #     fmt="g",
    #     cmap=cmap,
    #     vmin=1,
    #     vmax=10,
    #     ax=ax,
    #     cbar=show_cbar,
    #     cbar_ax=cbar_ax if show_cbar else None,
    # )
    # cbar_ax.tick_params(labelsize=10)

    sns.heatmap(
        pivot_table,
        fmt="g",
        cmap=plt.get_cmap("Greys_r"),
        vmin=1,
        vmax=10,
        ax=ax,
        cbar=show_cbar,
        cbar_ax=cbar_ax if show_cbar else None,
        linecolor="black",
        linewidths=0.5
    )
    # if show_cbar:
        # cbar_ax.set_ylabel("Score", fontsize=11)s
    cbar_ax.tick_params(labelsize=10)


    # ax.set_xticklabels([f"{int(x)//1024}k" for x in pivot_table.columns], rotation=45, fontsize=8)
    ax.set_xticklabels(['4k', '7k', '10k', '13k', '16k', '19k', '22k', '25k', '28k', '32k'], rotation=90, fontsize=11)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x*10:.0f}'))  # Format y-axis as percentage
    ax.tick_params(axis='y', labelrotation=0, labelsize=11)
    ax.tick_params(axis='x', labelrotation=45, labelsize=11)

import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--shadowkv-t-result', type=str, required=True)
    parser.add_argument('--loki-t-result', type=str, required=True)
    parser.add_argument('--kvswap-t-result', type=str, required=True)
    parser.add_argument('--output-pdf', type=str, required=True)
    args = parser.parse_args()

    result_dir_list = [args.shadowkv_t_result, args.loki_t_result, args.kvswap_t_result]
    output_path = args.output_pdf

    n_rows = 1
    n_cols = 3

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(7.5, 2)) 
    axs = axs.reshape(n_rows, n_cols)

    fig.subplots_adjust(
        left=0.0, right=0.88, top=1, bottom=0.0,
        hspace=0.25, wspace=0.15  
    )
    cbar_ax = fig.add_axes([0.89, 0.15, 0.015, 0.7])

    for i, result_json in enumerate(result_dir_list):
        row = i // n_cols
        col = i % n_cols
        ax = axs[row, col]
        
        draw(ax=ax, result_json=result_json, show_cbar=(i == 0), cbar_ax=cbar_ax)

        if i == 0:    
            ax.set_title('ShadowKV-t', fontsize=11, fontweight='bold', pad=0)
            ax.set_ylabel('Depth Percent (%)', fontsize=13, labelpad=0)
            label = ax.set_xlabel('Token Limit', labelpad=0)
            label.set_fontsize(13)
        elif i == 1:
            ax.set_title('Loki-t', fontsize=11, fontweight='bold', pad=0)
            ax.set_xlabel('Token Limit', fontsize=13, labelpad=0)
            ax.yaxis.label.set_visible(False)
        elif i == 2:
            ax.set_title('KVSwap-t(NVMe)', fontsize=11, fontweight='bold', pad=0)
            ax.yaxis.label.set_visible(False)
            ax.set_xlabel('Token Limit', fontsize=13, labelpad=0)

    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.05)

