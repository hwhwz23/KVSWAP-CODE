from metrics import needle_score, string_match_part, multi_number, multi_words
import argparse
import os
import json
import numpy as np

METRICS_FN = {
    'niah': needle_score,
    'multi': multi_number,
    'vt': multi_words,
    'cwe': multi_words,
    'fwe': multi_words,
    'qa': string_match_part,
}

def get_metric(filename):
    if 'multiquery' in filename or 'multivalue' in filename:
        return METRICS_FN['multi']
    elif 'niah' in filename:
        return METRICS_FN['niah']
    elif 'vt' in filename:
        return METRICS_FN['vt']
    elif 'cwe' in filename:
        return METRICS_FN['cwe']
    elif 'fwe' in filename:
        return METRICS_FN['fwe']
    elif 'qa' in filename:
        return METRICS_FN['qa']
    else:
        raise Exception("Metric not found")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RULER evaluation script")
    parser.add_argument('--results_dir', type=str)
    parser.add_argument('--prefix', type=str)
    args = parser.parse_args()
    
    results_dir = args.results_dir
    prefix = args.prefix
    
    files = os.listdir(results_dir)
    files = [file for file in files if prefix in file]
    # sort files by name
    files.sort()
    
    scores = {}
    avg_scores_list = []
    for file in files:
        if not file.endswith(".jsonl"):
            continue
        if 'fwe' in file:
            continue
        print(file)
        metric = get_metric(file)
        # ['niah_single_1', 'niah_single_2', 'niah_multikey_1', 'niah_multikey_2', 
                        #    'niah_multiquery', 'niah_multivalue', 'vt', 'fwe', 'qa_1', 'qa_2']
        if 'niah_single_1' in file:
            dataset_name = 'niah_single_1'
        elif 'niah_single_2' in file:
            dataset_name = 'niah_single_2'
        elif 'niah_multikey_1' in file:
            dataset_name = 'niah_multikey_1'
        elif 'niah_multikey_2' in file:
            dataset_name = 'niah_multikey_2'        
        elif 'niah_multiquery' in file: 
            dataset_name = 'niah_multiquery'
        elif 'niah_multivalue' in file:
            dataset_name = 'niah_multivalue'
        elif 'vt' in file:
            dataset_name = 'vt'
        elif 'fwe' in file:
            dataset_name = 'fwe'
        elif 'qa_1' in file:
            dataset_name = 'qa_1'
        elif 'qa_2' in file:
            dataset_name = 'qa_2'
        else:
            raise Exception("Gen len not found")
        
        filename = os.path.join(results_dir, file)
        scores_list = []
        input_lens, total_resp_tokens, resp_tokens, cot_tokens = [], [], [], []
        total_tokens = []

        with open(f"{filename}", "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                gt = data['gt']
                if isinstance(gt, list):
                    if len(gt) == 1:
                        gt = gt[0]
                scores_list.append(metric(data["response"], gt))
                
                if "cot_tokens" in data:
                    cot_t = data["cot_tokens"]
                else:
                    cot_t = 0
                cot_tokens.append(cot_t)
                input_lens.append(data["input_len"])
                total_resp_tokens.append(data["total_resp_tokens"])
                if data["resp_tokens"] == 0:
                    print(f"resp_tokens is 0 in {data['_id']}")
                # print(f"resp: {data['response_cot']}")
                resp_tokens.append(data["resp_tokens"])
                total_tokens.append(data["input_len"] + data["total_resp_tokens"])
            
        avg_scores_list.append(np.mean(scores_list))
        scores[dataset_name] = round(np.mean(scores_list)*100, 2)
        scores[dataset_name + '_num'] = len(scores_list)
        scores[dataset_name + '_input_lens'] = np.mean(input_lens).round()
        scores[dataset_name + '_total_resp_tokens'] = np.mean(total_resp_tokens).round()
        scores[dataset_name + '_resp_tokens'] = np.mean(resp_tokens).round()
        scores[dataset_name + '_cot_tokens'] = np.mean(cot_tokens).round()
        scores[dataset_name + '_total_tokens'] = np.mean(total_tokens).round()
    
    avg_score = np.mean(avg_scores_list)
    scores['avg'] = round(avg_score * 100, 2)
    scores['avg_num'] = sum([v for k, v in scores.items() if 'num' in k])
    scores['avg_input_lens'] = round(np.mean([v for k, v in scores.items() if 'input_lens' in k]), 2)
    scores['avg_total_resp_tokens'] = round(np.mean([v for k, v in scores.items() if 'total_resp_tokens' in k]), 2)
    scores['avg_resp_tokens'] = round(np.mean([v for k, v in scores.items() if 'resp_tokens' in k]), 2)
    scores['avg_cot_tokens'] = round(np.mean([v for k, v in scores.items() if 'cot_tokens' in k]), 2)
    scores['avg_total_tokens'] = round(np.mean([v for k, v in scores.items() if 'total_tokens' in k]), 2)
    sorted_scores = dict(sorted(scores.items(), key=lambda item: item[0]))
    
    if len(sorted_scores) > 0:
        with open(f"{results_dir}/results_{prefix}.json", "w") as f:
            json.dump(sorted_scores, f, ensure_ascii=False, indent=4)
    else:
        print("No results found")