import yaml
import os
import json
import re

import time
import requests
from tqdm import tqdm

import threading
from queue import Queue

import argparse

api_key = os.getenv('DS_API_KEY')
if api_key is None:
    raise ValueError("DS_API_KEY is not set")

def pred_openai(model_name, msg, base_url='https://api.openai.com'):
    tries = 0
    while tries < 5:
        tries += 1
        try:
            headers = {
                'Authorization': f"Bearer {api_key}"
            }
            resp = requests.post(f"{base_url}/v1/chat/completions", json = {
                "model": model_name,
                "messages": msg,
                "temperature": 0.
            }, headers=headers, timeout=120)
            if resp.status_code != 200:
                raise Exception(resp.text)
            resp = resp.json()
            break
        except KeyboardInterrupt as e:
            raise e
        except Exception as e:
            if "maximum context length" in str(e):
                raise e
            print("Error Occurs: \"%s\"        Retry ..."%(str(e)))
            time.sleep(1)
    else:
        print("Max tries. Failed.")
        return
    
    return resp["choices"][0]["message"]["content"]


USER_TEMPLATE = '''[Instruction]\nPlease act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. {criteria}[Ground truth]\n{reference}\nBegin your evaluation by providing a short explanation. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following this format: "[[rating]]", for example: "Rating: [[5]]".\n\n[Question]\n{input}\n\n[The Start of Assistant\'s Answer]\n{prediction}\n[The End of Assistant\'s Answer]'''
SYSTEM_TEMPLATE = 'You are a helpful assistant.'
CRITERIA = {
    "accuracy": """
    Score 1: The answer is completely unrelated to the reference.
    Score 3: The answer has minor relevance but does not align with the reference.
    Score 5: The answer has moderate relevance but contains inaccuracies.
    Score 7: The answer aligns with the reference but has minor omissions.
    Score 10: The answer is completely accurate and aligns perfectly with the reference.
    Only respond with a numberical score
    """
}

def get_criteria():
    cri = 'For this evaluation, you should primarily consider the following criteria:\n'
    for key, value in CRITERIA.items():
        cri += f'{key}: {value}\n'

    return cri

def get_user_template(input, prediction, reference, criteria):
    return USER_TEMPLATE.format(
        input=input,
        prediction=prediction,
        reference=reference,
        criteria=criteria
    )

if __name__ == '__main__':
    
    def eval(pred_jsonl):
        save_dir = os.path.dirname(pred_jsonl)
        # model_name = "gpt-4o-mini"
        # model_provider = "OpenAI"
        model_name = "deepseek-chat"
        model_provider = "DeepSeek"
        
        criteria = get_criteria()
        reference = "\nThe best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day.\n"
        input = "What is the best thing to do in San Francisco?"

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        out_file = f'{save_dir}/{os.path.basename(pred_jsonl).split(".jsonl")[0]}-{model_provider}_{model_name}.jsonl'

        has_data = {}
        if os.path.exists(out_file):
            with open(out_file, encoding='utf-8') as f:
                has_data = {json.loads(line)["_id"]: 0 for line in f}
        fout = open(out_file, 'a', encoding='utf-8')
        data = []
        data_all = [json.loads(line) for line in open(pred_jsonl, encoding='utf-8')]
        print(f"Total {len(data_all)} data loaded")
        
        for item in data_all:
            if item["_id"] not in has_data:
                data.append(item)    

        print(f"Total {len(data)} data to eval")

        def process(data_sub, out_queue):
            for item in tqdm(data_sub):
                user_template = get_user_template(input, item['response'], reference, criteria)

                if model_provider in ('OpenAI', "DeepSeek"):
                    msg = [{
                            "role": "system",
                            "content": SYSTEM_TEMPLATE
                        }, {
                            "role": "user",
                            "content": user_template
                        }
                    ]
                    base_url = 'https://api.openai.com' if model_provider == 'OpenAI' else 'https://api.deepseek.com'
                    result = pred_openai(model_name, msg, base_url=base_url)
                else:
                    raise NotImplementedError(f'Not implemented model provider: {model_provider}')
            
                pattern = r"\[\[(\d+)\]\]"
                match = re.search(pattern, result)
                score = int(match.group(1)) if match else None
                item['score'] = score
                item['eval_result'] = result
                out_queue.put(item)
        
        def write_to_file(out_queue):
            while True:
                item = out_queue.get()
                if item is None:
                    print("Writer thread exit.")
                    break
                if 'score' not in item:
                    print(f"Error: {item}")
                    continue
                fout.write(json.dumps(item, ensure_ascii=False) + '\n')
                fout.flush()
                
        out_queue = Queue()
        writer_thread = threading.Thread(target=write_to_file, args=(out_queue,))
        writer_thread.start()
        
        split_num = 8
        data_split = [data[i::split_num] for i in range(split_num)]
        threads = []
        for i in range(split_num):
            t = threading.Thread(target=process, args=(data_split[i], out_queue))
            threads.append(t)
            t.start()
        for t in threads:
            t.join()
        out_queue.put(None)
        writer_thread.join()
        fout.flush()
        fout.close()
        print("Done.")
        print(f"Results saved to {out_file}")
        
    def eval_dir(jsonl_dir):    
        for pred_jsonl in os.listdir(jsonl_dir):
            if not pred_jsonl.endswith('.jsonl'):
                continue
            if 'deepseek' in pred_jsonl.lower() or 'openai' in pred_jsonl.lower():
                continue
            print(pred_jsonl)
            eval(os.path.join(jsonl_dir, pred_jsonl))
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--jsonl_dir', type=str)
    args = parser.parse_args()
    jsonl_dir = args.jsonl_dir
    if not os.path.exists(jsonl_dir):
        raise ValueError(f"Directory {jsonl_dir} does not exist.")
    eval_dir(jsonl_dir)

    