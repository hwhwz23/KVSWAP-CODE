import os
import json
import argparse
import numpy as np

from LongBench.longbenchv1_metrics import (
    qa_f1_score,
    rouge_zh_score,
    qa_f1_zh_score,
    rouge_score,
    classification_score,
    retrieval_score,
    retrieval_zh_score,
    count_score,
    code_sim_score,
)

dataset2metric = {
    "narrativeqa": qa_f1_score,
    "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    "multifieldqa_zh": qa_f1_zh_score,
    "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "musique": qa_f1_score,
    "dureader": rouge_zh_score,
    "gov_report": rouge_score,
    "qmsum": rouge_score,
    "multi_news": rouge_score,
    "vcsum": rouge_zh_score,
    "trec": classification_score,
    "triviaqa": qa_f1_score,
    "samsum": rouge_score,
    "lsht": classification_score,
    "passage_retrieval_en": retrieval_score,
    "passage_count": count_score,
    "passage_retrieval_zh": retrieval_zh_score,
    "lcc": code_sim_score,
    "repobench-p": code_sim_score,
}

def scorer_e(dataset, predictions, answers, lengths, all_classes):
    scores = {"0-4k": [], "4-8k": [], "8k+": []}
    for (prediction, ground_truths, length) in zip(predictions, answers, lengths):
        score = 0.
        if 'trec' in dataset or 'triviaqa' in dataset or 'samsum' in dataset or 'lsht' in dataset:
            prediction = prediction.lstrip('\n').split('\n')[0]
        for ground_truth in ground_truths:
            score = max(score, dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))
        if length < 4000:
            scores["0-4k"].append(score)
        elif length < 8000:
            scores["4-8k"].append(score)
        else:
            scores["8k+"].append(score)
    sample_nums = {}
    for key in scores.keys():
        sample_nums[key] = len(scores[key])
        scores[key] = round(100 * np.mean(scores[key]), 2)
    sample_nums["all"] = sum(sample_nums.values())
    return scores, sample_nums

def scorer(dataset, predictions, answers, all_classes):
    total_score = 0.
    for (prediction, ground_truths) in zip(predictions, answers):
        score = 0.
        if 'trec' in dataset or 'triviaqa' in dataset or 'samsum' in dataset or 'lsht' in dataset:
            prediction = prediction.lstrip('\n').split('\n')[0]
        for ground_truth in ground_truths:
            score = max(score, dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))
        total_score += score
    return round(100 * total_score / len(predictions), 2)


def result_v2(files, results_dir):
    output = ["Model\tOverall\tEasy\tHard\tShort\tMedium\tLong"]
    compensated = True

    for file in files:
        if not file.endswith('.jsonl'):
            continue
        print(file)
        filename = os.path.join(results_dir, file)
        try:
            pred_data = json.load(open(filename, encoding='utf-8'))
        except Exception as e:
            try:
                pred_data = [json.loads(line) for line in open(filename, encoding='utf-8')]
            except:
                print(f"Error loading {filename}: {e}")
                continue
        easy, hard, short, medium, long = 0, 0, 0, 0, 0
        easy_acc, hard_acc, short_acc, medium_acc, long_acc = 0, 0, 0, 0, 0
        for pred in pred_data:
            acc = int(pred['judge'])
            if compensated and pred["pred"] == None:
                acc = 0.25
            if pred["difficulty"] == "easy":
                easy += 1
                easy_acc += acc
            else:
                hard += 1
                hard_acc += acc

            if pred['length'] == "short":
                short += 1
                short_acc += acc
            elif pred['length'] == "medium":
                medium += 1
                medium_acc += acc
            else:
                long += 1
                long_acc += acc

        print(easy, hard, short, medium, long)
        print(easy_acc, hard_acc, short_acc, medium_acc, long_acc)
        name = '.'.join(file.split('.')[:-1])
        overall_acc = round(100*(easy_acc+hard_acc)/len(pred_data), 1)
        try:
            easy_acc = round(100*easy_acc/easy, 1)
        except:
            easy_acc = 'nan'
        try:
            hard_acc = round(100*hard_acc/hard, 1)
        except:
            hard_acc = 'nan'
        try:
            short_acc = round(100*short_acc/short, 1)
        except:
            short_acc = 'nan'
        try:
            medium_acc = round(100*medium_acc/medium, 1)
        except:
            medium_acc = 'nan' 
        try:
            long_acc = round(100*long_acc/long, 1)
        except:
            long_acc = 'nan'
        
        output.append(name+'\t'+str(overall_acc)+'\t'+str(easy_acc)+'\t'+str(hard_acc)+'\t'+str(short_acc)+'\t'+str(medium_acc)+'\t'+str(long_acc))

    open(f'{results_dir}/result.txt', 'w', encoding='utf-8').write('\n'.join(output))


task_type_map = {
    "narrativeqa": 'SQA',
    "qasper": 'SQA',
    "multifieldqa_en": 'SQA',
    # "multifieldqa_zh": qa_f1_zh_score,
    "hotpotqa": 'MQA',
    "2wikimqa": 'MQA',
    "musique": 'MQA',
    # "dureader": rouge_zh_score,
    "gov_report": 'SUM',
    "qmsum": 'SUM',
    "multi_news": 'SUM',
    # "vcsum": 'SUM',
    "trec": 'FSL',
    "triviaqa": 'FSL',
    "samsum": 'FSL',
    # "lsht": 'FSL',
    "passage_retrieval_en": 'SYN',
    "passage_count": 'SYN',
    # "passage_retrieval_zh": retrieval_zh_score,
    "lcc": 'CODE',
    "repobench-p": 'CODE',
}

def result(files, results_dir, prefix, is_cot):
    scores = dict()
    tasks_type_scores = dict()
    is_e = False
    
    if 'longbench-e' in results_dir:
        is_e = True
        print("Evaluating on longbench-e")
    
    for file in files:
        if not file.endswith(".jsonl"):
            continue
        if is_cot:
            if 'cot' not in file:
                continue
        else:
            if 'cot' in file:
                continue
        print(file)
        dataset_name = os.path.splitext(file)[0]
        for dataset in dataset2metric.keys():
            if dataset in dataset_name:
                dataset_name = dataset
                break
        
        filename = os.path.join(results_dir, file)
        predictions, answers, lengths = [], [], []
        input_lens, total_resp_tokens, resp_tokens, cot_tokens = [], [], [], []
        total_tokens = []

        with open(f"{filename}", "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                if data["input_len"] < 4096:
                    continue
                response = data["response"]
                # if 'response_cot' in data and len(data["response_cot"]) == 0:
                #     # response = response.split("\n")[-1]
                #     print("=="*20)
                #     print(response)
                #     print("=="*20)
                #     print(response.split("\n")[-1])
                #     print("=="*20)
                predictions.append(response)
                answers.append(data["answers"])
                all_classes = data["all_classes"]
                if "length" in data:
                    lengths.append(data["length"])
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
        scores[dataset_name + '_input_lens'] = np.mean(input_lens).round()
        scores[dataset_name + '_total_resp_tokens'] = np.mean(total_resp_tokens).round()
        scores[dataset_name + '_resp_tokens'] = np.mean(resp_tokens).round()
        scores[dataset_name + '_cot_tokens'] = np.mean(cot_tokens).round()
        scores[dataset_name + '_total_tokens'] = np.mean(total_tokens).round()

        if is_e:
            score, sample_nums = scorer_e(dataset_name, predictions, answers, lengths, all_classes)
            scores[dataset_name] = score
            scores[dataset_name+'_num'] = sample_nums
        else:
            try:
                score = scorer(dataset_name, predictions, answers, all_classes)
            except Exception as e:
                print(f"Error calculating score for {dataset_name}: {e}")
                exit(0)
            scores[dataset_name] = score
            scores[dataset_name+'_num'] = len(predictions)

        task_type = task_type_map[dataset_name]
        if task_type not in tasks_type_scores:
            tasks_type_scores[task_type] = []
            tasks_type_scores[task_type+'_input_lens'] = []
            tasks_type_scores[task_type+'_total_resp_tokens'] = []
            tasks_type_scores[task_type+'_resp_tokens'] = []
            tasks_type_scores[task_type+'_cot_tokens'] = []
            tasks_type_scores[task_type+'_total_tokens'] = []
            tasks_type_scores[task_type+'_num'] = []
             
        tasks_type_scores[task_type] += [score]
        tasks_type_scores[task_type+'_input_lens'] += [scores[dataset_name + '_input_lens']]
        tasks_type_scores[task_type+'_total_resp_tokens'] += [scores[dataset_name + '_total_resp_tokens']]
        tasks_type_scores[task_type+'_resp_tokens'] += [scores[dataset_name + '_resp_tokens']]
        tasks_type_scores[task_type+'_cot_tokens'] += [scores[dataset_name + '_cot_tokens']]
        tasks_type_scores[task_type+'_total_tokens'] += [scores[dataset_name + '_total_tokens']]
        tasks_type_scores[task_type+'_num'] += [scores[dataset_name+'_num']]        
    
    if is_cot:
        prefix += 'cot'
    
    avg_tasks_type_scores = {
        'score': [], 
        'input_lens': [],
        'total_resp_tokens': [],
        'resp_tokens': [],
        'cot_tokens': [],
        'total_tokens': [],
        'num': []   
    }
    
    for task_type in list(set(task_type_map.values())):
        if task_type not in tasks_type_scores:
            continue
        tasks_type_scores[task_type] = round(np.mean(tasks_type_scores[task_type]), 2)
        tasks_type_scores[task_type+'_input_lens'] = round(np.mean(tasks_type_scores[task_type+'_input_lens']), 2)
        tasks_type_scores[task_type+'_total_resp_tokens'] = round(np.mean(tasks_type_scores[task_type+'_total_resp_tokens']), 2)
        tasks_type_scores[task_type+'_resp_tokens'] = round(np.mean(tasks_type_scores[task_type+'_resp_tokens']), 2)
        tasks_type_scores[task_type+'_cot_tokens'] = round(np.mean(tasks_type_scores[task_type+'_cot_tokens']), 2)
        tasks_type_scores[task_type+'_total_tokens'] = round(np.mean(tasks_type_scores[task_type+'_total_tokens']), 2)
        tasks_type_scores[task_type+'_num'] = sum(tasks_type_scores[task_type+'_num'])    
        avg_tasks_type_scores['score'].append(tasks_type_scores[task_type])
        avg_tasks_type_scores['input_lens'].append(tasks_type_scores[task_type+'_input_lens'])
        avg_tasks_type_scores['total_resp_tokens'].append(tasks_type_scores[task_type+'_total_resp_tokens'])
        avg_tasks_type_scores['resp_tokens'].append(tasks_type_scores[task_type+'_resp_tokens'])
        avg_tasks_type_scores['cot_tokens'].append(tasks_type_scores[task_type+'_cot_tokens'])
        avg_tasks_type_scores['total_tokens'].append(tasks_type_scores[task_type+'_total_tokens'])
        avg_tasks_type_scores['num'].append(tasks_type_scores[task_type+'_num'])    

    # calculate the average score for each task type
    avg_tasks_type_scores['score'] = round(np.mean(avg_tasks_type_scores['score']), 2)
    avg_tasks_type_scores['input_lens'] = round(np.mean(avg_tasks_type_scores['input_lens']), 2)
    avg_tasks_type_scores['total_resp_tokens'] = round(np.mean(avg_tasks_type_scores['total_resp_tokens']), 2)
    avg_tasks_type_scores['resp_tokens'] = round(np.mean(avg_tasks_type_scores['resp_tokens']), 2)
    avg_tasks_type_scores['cot_tokens'] = round(np.mean(avg_tasks_type_scores['cot_tokens']), 2)
    avg_tasks_type_scores['total_tokens'] = round(np.mean(avg_tasks_type_scores['total_tokens']), 2)
    avg_tasks_type_scores['num'] = sum(avg_tasks_type_scores['num'])
    tasks_type_scores['AVG'] = avg_tasks_type_scores

    # sort the scores by dataset name
    sorted_scores = dict(sorted(scores.items(), key=lambda item: item[0]))
    sorted_tasks_type_scores = dict(sorted(tasks_type_scores.items(), key=lambda item: item[0])) 
    
    if len(sorted_scores) > 0:
        with open(f"{results_dir}/results_{prefix}.json", "w") as f:
            json.dump(sorted_scores, f, ensure_ascii=False, indent=4)

        with open(f"{results_dir}/tasks_results_{prefix}.json", "w") as f:
            json.dump(sorted_tasks_type_scores, f, ensure_ascii=False, indent=4)
    else:
        print(f"No results found in {results_dir} for prefix {prefix}. Please check the files.")


import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process longbench results.")
    parser.add_argument('--results_dir', type=str)
    parser.add_argument('--prefix', type=str)

    args = parser.parse_args()
    results_dir = args.results_dir
    prefix = args.prefix

    if not os.path.exists(results_dir):
        print(f"Results directory {results_dir} does not exist.")
        exit(0)

    files = os.listdir(results_dir)
    files = [file for file in files if prefix in file]
    # sort files by name
    files.sort()

    if 'longbenchv2' in results_dir:
        result_v2(files, results_dir)
    else:
        if 'cot' in results_dir:
            result(files, results_dir, prefix, is_cot=True)
        if not 'cot_only' in results_dir:
            result(files, results_dir, prefix, is_cot=False)
    
    