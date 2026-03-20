import torch
import numpy as np
import argparse
from tqdm import tqdm
import json
import os
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
from models.modeling_llama import eager_attention_forward as llama_eager_attention_forward
from models.modeling_llama import att_forward as llama_att_forward
from models.modeling_llama import model_forward as llama_model_forward

from models.modeling_qwen2_5_vl import att_forward as qwen2_5_vl_att_forward
from models.modeling_qwen2_5_vl import model_forward as qwen2_5_vl_model_forward

from models.modeling_qwen3 import eager_attention_forward as qwen3_eager_attention_forward
from models.modeling_qwen3 import att_forward as qwen3_att_forward
from models.modeling_qwen3 import model_forward as qwen3_model_forward

# from models.modeling_opt import att_forward as opt_att_forward
# from models.modeling_opt import decoder_forward as opt_decoder_forward
from utils import *
import gc 
import re
import glob
import hashlib
from template import apply_template as apply_template
from qwen_vl_utils import process_vision_info
from internvl3_utils import load_video as load_video_internvl3
import copy
import random   
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLModel
from shadowkv.models.base import LLM as ShadowKVLLM

class Catcher_Qwen2_5_VLModel(Qwen2_5_VLModel):
    def __init__(self, config):
        super(Catcher_Qwen2_5_VLModel, self).__init__(config)
    
    def reset_catcher(self):
        self.saved = []
    
    def forward(self, *args, **kwargs):
        self.saved.append((args, kwargs))
        # if len(self.saved) >= 5:
        raise Exception("Catcher called, saving video embeddings")
        # return super(Catcher_Qwen2_5_VLModel, self).forward(*args, **kwargs)

MIN_LEN = 4096

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

def clear_layer_stats(decoder_model):
    if decoder_model is None:
        return
    for layer_i in range(len(decoder_model.layers)):
        if hasattr(decoder_model.layers[layer_i].self_attn, 'fetch_num_list'):
            del decoder_model.layers[layer_i].self_attn.fetch_num_list
        if hasattr(decoder_model.layers[layer_i].self_attn, 'density_list'):
            del decoder_model.layers[layer_i].self_attn.density_list

def print_layer_stats(decoder_model, model):
    if decoder_model is None:
        model.print_kv_stats()
        print("===============================================")
        return
    layer_density = []
    layer_fetch_num = []
    for layer_i in range(len(decoder_model.layers)):
        if hasattr(decoder_model.layers[layer_i].self_attn, 'density_list'):
            layer_density.append(decoder_model.layers[layer_i].self_attn.density_list)
        if hasattr(decoder_model.layers[layer_i].self_attn, 'fetch_num_list'):
            layer_fetch_num.append(decoder_model.layers[layer_i].self_attn.fetch_num_list)
            
    if len(layer_density) > 0:
        layer_density = torch.tensor(layer_density).float() # l,n
        layer_fetch_num = torch.tensor(layer_fetch_num).float() # l,n      
        layer_density = layer_density.mean(dim=1) # l
        layer_fetch_num = layer_fetch_num.mean(dim=1) # l
        if args.start_layer > 0:
            layer_density = torch.cat((torch.ones(args.start_layer), layer_density), dim=0)
        layer_density = layer_density.cpu().numpy().round(4)
        print("layer density: ", layer_density.tolist())
        avg_density = layer_density.mean().item()
        print(f"avg density: {avg_density:.4f}")
        layer_fetch_num = layer_fetch_num.cpu().numpy().round()
        print(f"layer_fetch_num: ", layer_fetch_num.tolist())
        avg_fetch_num = layer_fetch_num.mean().item()
        print(f"avg fetch num: {avg_fetch_num:.1f}")
    print("===============================================")
    

def query_llm(args, id, prompt, model, tokenizer, enable_cot, gen_configs, task):

    max_len = args.seq_len - gen_configs.max_new_tokens
    input_ids = tokenizer.encode(prompt)
    
    if len(input_ids) > max_len:
        print(f"{id} prompt too long: {len(input_ids)} > {max_len}", flush=True)
        input_ids = input_ids[:max_len//2] + input_ids[-max_len//2:]
        prompt = tokenizer.decode(input_ids, skip_special_tokens=True)
    elif len(input_ids) < MIN_LEN and ('needle' not in task and 'livecodebench' not in task and 'math500' not in task):
        print(f'{id} prompt too short, skipping: {len(input_ids)} < {MIN_LEN}', flush=True)
        return None

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    gc.collect()
    torch.cuda.empty_cache()


    if isinstance(model, ShadowKVLLM):
        generated_ids = model.generate(input_ids=inputs.input_ids, gen_len=gen_configs.max_new_tokens, 
                    verbose=False, top_p=gen_configs.top_p, top_k=gen_configs.top_k, temperature=gen_configs.temperature)
        generated_ids = [generated_ids]
    else:
        extra_kwargs = {}
        if args.model_type in ('llama', 'ds_qwen3', 'ds_llama'):
            extra_kwargs['pad_token_id'] = tokenizer.eos_token_id

        with torch.no_grad():
            generated_ids  = model.generate(
                **inputs,
                generation_config=gen_configs,
                **extra_kwargs
            )
    if isinstance(generated_ids, list):
        output_ids = generated_ids[0]
    else:
        output_ids = generated_ids[0][len(inputs.input_ids[0]):].tolist() 
        
    thinkend_token = tokenizer.encode('</think>')[-1]
    # print("thinkend_token=", thinkend_token)
    
    if args.model_type in ('qwen3', 'ds_qwen3', 'ds_llama'):
        # parsing thinking content
        try:
            # rindex finding 151668 (</think>)
            index = len(output_ids) - output_ids[::-1].index(thinkend_token)
        except ValueError:
            index = 0
        thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")

    else:
        thinking_content = ""
        index = 0
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
    out_token_lens = (index, len(output_ids)-index)
    input_len = len(inputs.input_ids[0])
        
    return thinking_content, content, input_len, out_token_lens

def extract_answer(response):
    response = response.replace('*', '')
    match = re.search(r'The correct answer is \(([A-D])\)', response)
    if match:
        return match.group(1)
    else:
        match = re.search(r'The correct answer is ([A-D])', response)
        if match:
            return match.group(1)
        else:
            return None

def get_pred(args, model, data, tokenizer, mode, fout, task, gen_configs, out_file, **kwargs):
    print(f"Processing {len(data)} samples in {mode} mode")
    
    processed_samples = 0
    for item in tqdm(data):
        if 'longbench' in task:
            prompt_format = kwargs['prompt_format']
            prompt = prompt_format.format(**item)
            max_new_tokens = kwargs['max_gen']
            if mode == 'cot':
                max_new_tokens += 1000
            # ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]
            if 'trec' in out_file or 'triviaqa' in out_file or 'samsum' in out_file or 'lsht' in out_file or 'lcc' in out_file or 'repobench-p' in out_file:
                assert mode != 'cot', f"mode {mode} not supported for task {task}"
                prompt = prompt
            else:
                prompt = apply_template(args, prompt, tokenizer, mode == 'cot')
        elif 'ruler' in task:
            prompt = item['content']
            if mode == 'cot':
                raise NotImplementedError
            else:
                max_new_tokens = kwargs['max_gen']
                if args.model_type == 'qwen3':
                    prompt += '<think>\n\n</think>\n\n'
                answer_prefix = item['answer_prefix']
                if len(answer_prefix) > 0:
                    prompt += answer_prefix      
        elif 'needle' in task:
            prompt = item['content']
            prompt = apply_template(args, prompt, tokenizer, mode == 'cot')
            if mode == 'cot':
                raise NotImplementedError
            else:
                max_new_tokens = 150
        else:
            raise ValueError(f"task {task} not supported")

        gen_configs.max_new_tokens = max_new_tokens
        
        query_result = query_llm(args, item["_id"], prompt, model, tokenizer, mode == 'cot', gen_configs, task)
        if query_result is None:
            print(f"Skipping item {item['_id']}", flush=True)
            continue
        thinking_content, content, input_len, out_token_lens = query_result
        thinking_content = thinking_content.strip()
        content = content.strip()
        thinking_len, content_len = out_token_lens

        if mode == 'cot':
            item['response_cot'] = thinking_content
            item['cot_tokens'] = thinking_len
            if content_len == 0:
                print(f"Empty response for {item['_id']}, max_new_tokens not enough", flush=True)
        else:
            # assert thinking_len == 0, f"thinking_content {thinking_content} not empty, len={thinking_len}"
            if thinking_len > 0:
                print(f"thinking_content {thinking_content} not empty, len={thinking_len}, id={item['_id']}", flush=True)

        response = content
        item['response'] = response
        item['input_len'] = input_len
        item['total_resp_tokens'] = content_len + thinking_len
        item['resp_tokens'] = content_len
        item['prompt'] = prompt[len(prompt)-64:]
        if 'needle' in task:
            del item['content']
        elif 'ruler' in task:
            del item['content']
        fout.write(json.dumps(item, ensure_ascii=False) + '\n')
        fout.flush()

        processed_samples += 1
        if args.eval_samples > 0 and processed_samples >= args.eval_samples:
            print(f"Processed {processed_samples} samples, reached eval_samples={args.eval_samples}", flush=True)
            break


def process_out_file_list(args, model, decoder_model, out_file_list, data_all, task, gen_configs, tokenizer, **kwargs):

    thinkend_token = tokenizer.encode('</think>')[-1]
    print("thinkend_token=", thinkend_token, flush=True)

    for out_file in out_file_list:
        has_data = {}
        print(f"Loading {out_file}...")
        args.eval_samples = int(args.eval_samples)
        assert args.eval_samples == 0 or args.eval_samples > 1, f"args.eval_samples={args.eval_samples}"

        if os.path.exists(out_file):
            with open(out_file, encoding='utf-8') as f:
                has_data = {json.loads(line)["_id"]: 0 for line in f}
        
        if args.eval_samples > 0 and len(has_data) >= args.eval_samples:
            print(f"All samples in {out_file} are already processed, reached eval_samples={args.eval_samples}, skipping")
            continue
        elif args.eval_samples > 0:
            args.eval_samples = args.eval_samples - len(has_data)

        fout = open(out_file, 'a', encoding='utf-8')
        data = []

        for item in data_all:
            if item["_id"] not in has_data:
                data.append(item)
        if len(data) == 0:
            print(f"All samples in {out_file} are already processed")
            continue
        else:
            print(f"Processing {len(data)} samples in {out_file}")
        clear_layer_stats(decoder_model)
        get_pred(args, model, data, tokenizer, 'cot' if '_cot.jsonl' in out_file else 'gen', fout, task, gen_configs, out_file, **kwargs)
        fout.close()
        print(f"Saved to {out_file}")
        print_layer_stats(decoder_model, model)

def inference_single_video_internvl3(args, vid_path, sys_ins, inp, model, tokenizer, max_new_tokens, gen_configs, PROCESSED_DATA_DIR, DO_REUSE_PROCESSED, DO_SAVE_PROCESSED):
    assert args.model_type == 'internvl3', f"Model type {args.model_type} not supported for video input"
    assert sys_ins == '', f"System instruction {sys_ins} not supported for now"
    # MAX_NUM_SEGMENTS = 64
    MAX_NUM_SEGMENTS = 128
    # MAX_NUM_SEGMENTS = 180
    # MAX_NUM_SEGMENTS = 256

    def hash_messages(messages):
        return hashlib.md5(json.dumps(messages, sort_keys=True).encode('utf-8')).hexdigest()
    
    if DO_REUSE_PROCESSED or DO_SAVE_PROCESSED:
        msg_hash = hash_messages({'video': os.path.basename(vid_path), 'inp': inp, 'max_num_segments': MAX_NUM_SEGMENTS})
    
    if DO_REUSE_PROCESSED and os.path.exists(f"{PROCESSED_DATA_DIR}/{msg_hash}.pt"):
        question, pixel_values, num_patches_list = torch.load(f"{PROCESSED_DATA_DIR}/{msg_hash}.pt", weights_only=False)
        print(f"Loaded preprocessed inputs for {vid_path} from {PROCESSED_DATA_DIR}/{msg_hash}.pt", flush=True)
    else:
        if hasattr(args, 'last_loaded_video') and args.last_loaded_video_path == vid_path:
            print(f"Reusing last loaded video {args.last_loaded_video_path}", flush=True)
            pixel_values_, num_patches_list_ = args.last_loaded_video
            pixel_values = pixel_values_.clone()
            num_patches_list = copy.deepcopy(num_patches_list_)
        else:   
            pixel_values, num_patches_list = load_video_internvl3(vid_path, input_size=448, num_segments=MAX_NUM_SEGMENTS, max_num=1)
            pixel_values = pixel_values.to(args.torch_dtype).to(model.device)
            args.last_loaded_video = (pixel_values.clone(), copy.deepcopy(num_patches_list))
            args.last_loaded_video_path = vid_path
        # Frame1: <image>\nFrame2: <image>\n...\nFrame8: <image>\n{question}
        video_prefix = ''.join([f'Frame-{i+1}: <image>\n' for i in range(len(num_patches_list))])
        question = video_prefix + inp
    
    if DO_SAVE_PROCESSED:
        print(f"Saving preprocessed inputs for {vid_path} to {PROCESSED_DATA_DIR}/{msg_hash}.pt", flush=True)
        torch.save((question, pixel_values, num_patches_list), f"{PROCESSED_DATA_DIR}/{msg_hash}.pt")
    # generation_config = dict(max_new_tokens=max_new_tokens, do_sample=True, pad_token_id=tokenizer.eos_token_id)    
    gen_configs.max_new_tokens = max_new_tokens
    response, generation_output, input_ids = model.chat(tokenizer, pixel_values, question, gen_configs,
                            num_patches_list=num_patches_list, history=None, return_history=False)
    inputs_meta = {
        'input_tok_len': len(input_ids[0]),
    }
    
    return response, inputs_meta, len(generation_output[0])

def inference_single_video(args, vid_path, sys_ins, inp, model, processor, max_new_tokens, gen_configs, 
                           PROCESSED_DATA_DIR, DO_REUSE_PROCESSED, DO_SAVE_PROCESSED, total_pixels=24576 * 28 * 28):

    torch.cuda.empty_cache()
    
    # total_pixels = 24576 * 28 * 28
    min_pixels = 16 * 28 * 28
    
    if len(sys_ins) > 0:
        messages = [{"role": "system", "content": sys_ins}, ]
    else:
        messages = []
    
    if args.model_type == 'qwen2.5vl':
        user_msg = [
            {"role": "user", "content": [
                {'type': 'video', 'video': vid_path, 'total_pixels': total_pixels, 'min_pixels': min_pixels},
                {"type": "text", "text": inp},]
            },
        ]
    elif args.model_type == 'videollama3':
        user_msg = [
            {"role": "user", "content": [
                {"type": "video", "video": {"video_path": vid_path, "fps": 1, "max_frames": 180}},
                {"type": "text", "text": inp},]
            },    
        ]
    else:
        raise NotImplementedError(f"Model type {args.model_type} not supported for video input")
    
    messages.extend(user_msg)   

    def hash_messages(messages):
        """Hash the messages to create a unique identifier."""
        vid_path_ = messages[-1]['content'][0]['video']
        if isinstance(vid_path_, dict):
            vid_path_ = os.path.basename(vid_path_['video_path'])
            messages[-1]['content'][0]['video']['video_path'] = vid_path_  # Only keep the video name
        else:
            messages[-1]['content'][0]['video'] = os.path.basename(vid_path_)
        return hashlib.md5(json.dumps(messages, sort_keys=True).encode('utf-8')).hexdigest()

    if DO_REUSE_PROCESSED or DO_SAVE_PROCESSED or args.save_video_emb or isinstance(model, ShadowKVLLM):
        msg_hash = hash_messages(messages)
        if isinstance(messages[-1]['content'][0]['video'], dict):
            messages[-1]['content'][0]['video']['video_path'] = vid_path
        else:
            messages[-1]['content'][0]['video'] = vid_path
        
    if DO_REUSE_PROCESSED and os.path.exists(f"{PROCESSED_DATA_DIR}/{msg_hash}.pt") and not isinstance(model, ShadowKVLLM):
        inputs, inputs_meta = torch.load(f"{PROCESSED_DATA_DIR}/{msg_hash}.pt", weights_only=False)
        print(f"Loaded preprocessed inputs for {vid_path} from {PROCESSED_DATA_DIR}/{msg_hash}.pt", flush=True)
        try:
            input_ids = inputs.input_ids
        except: 
            input_ids = inputs['input_ids']
    elif isinstance(model, ShadowKVLLM):
        VIDEO_EMB_DATA_DIR = f'./bench/MLVU/MVLU_DATA/{args.model_name}_video_embs'
        saved = torch.load(f"{VIDEO_EMB_DATA_DIR}/{msg_hash}.pt", weights_only=False)
        input_embs = saved[0][1]['inputs_embeds']
        position_ids = saved[0][1]['position_ids']
        rope_deltas = saved[0][1]['rope_deltas']
        cache_position = saved[0][1]['cache_position']
        inputs_meta = {
            'video_emb_shape': input_embs.shape,
        }
    else:
        if args.model_type == 'qwen2.5vl':
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs, video_kwargs = process_vision_info([messages], return_video_kwargs=True)
            fps_inputs = video_kwargs['fps']
            # print("video input:", video_inputs[0].shape, flush=True)
            num_frames, _, resized_height, resized_width = video_inputs[0].shape
            len_video_tokens = int(num_frames / 2 * resized_height / 28 * resized_width / 28)
            # print("fps inputs:", fps_inputs, flush=True)
            # print("num of video tokens:", len_video_tokens, flush=True)
            inputs = processor(text=[text], images=image_inputs, videos=video_inputs, fps=fps_inputs, padding=True, return_tensors="pt")
            inputs = inputs.to(model.device)
            inputs_meta = {
                'video_tok_len': len_video_tokens,
                'video_shapes_fps': (video_inputs[0].shape, fps_inputs)
            }
            input_ids = inputs.input_ids
        elif args.model_type == 'videollama3':
            inputs = processor(conversation=messages, 
                                add_system_prompt=True, 
                                add_generation_prompt=True,
                                return_tensors="pt")
            inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(args.torch_dtype)
            inputs_meta = {}
            input_ids = inputs['input_ids']
        else:
            raise NotImplementedError(f"Model type {args.model_type} not supported for video input")
        inputs_meta.update({
            'input_tok_len': len(input_ids[0]),
        })
        if DO_SAVE_PROCESSED:
            print(f"Saving preprocessed inputs for {vid_path} to {PROCESSED_DATA_DIR}/{msg_hash}.pt", flush=True)
            torch.save((inputs, inputs_meta), f"{PROCESSED_DATA_DIR}/{msg_hash}.pt")
    
    gen_configs.max_new_tokens = max_new_tokens
    
    if hasattr(model, "model") and isinstance(model.model, Catcher_Qwen2_5_VLModel):
        assert args.save_video_emb, "Catcher should only be used when saving video embeddings"
        assert not isinstance(model, ShadowKVLLM), "Catcher should not be used with ShadowKVLLM"
        VIDEO_EMB_DATA_DIR = f'./bench/MLVU/MVLU_DATA/{args.model_name}_video_embs'
        if not os.path.exists(VIDEO_EMB_DATA_DIR):
            os.makedirs(VIDEO_EMB_DATA_DIR, exist_ok=True)
        model.model.reset_catcher()
        try:
            generated_ids = model.generate(**inputs, generation_config=gen_configs)
        except Exception as e:
            print(f"Exception occurred during generation: {e}", flush=True)
            saved = model.model.saved
            saved[0][1]['rope_deltas'] = model.rope_deltas
            print(f"Catcher called, saving video embeddings for {vid_path}", flush=True)
            torch.save(saved, f"{VIDEO_EMB_DATA_DIR}/{msg_hash}.pt")
        return 'NULL', inputs_meta, 0 
    else:
        assert not args.save_video_emb, "Model should not be a Catcher when saving video embeddings"
        if isinstance(model, ShadowKVLLM):
            generated_ids = model.generate(input_ids=None, input_embs=input_embs, 
                                           gen_len=gen_configs.max_new_tokens, verbose=False, top_p=gen_configs.top_p, 
                                           top_k=gen_configs.top_k, temperature=gen_configs.temperature, 
                                            position_ids=position_ids, rope_deltas=rope_deltas, cache_position=cache_position)
            generated_ids_trimmed = [generated_ids]
        else:
            generated_ids = model.generate(**inputs, generation_config=gen_configs)
            if args.model_type == 'qwen2.5vl':
                generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(input_ids, generated_ids)]
            elif args.model_type == 'videollama3':
                generated_ids_trimmed = generated_ids
            else:
                raise NotImplementedError(f"Model type {args.model_type} not supported for video input")
        
        output = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return output, inputs_meta, len(generated_ids_trimmed[0])


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--model_path', type=str)
    parser.add_argument('--tasks', type=str)
    parser.add_argument('--seq_len', type=int)
    parser.add_argument('--eval_samples', type=float, default=0)

    parser.add_argument("--skewing_matrix_path", type=str, default=None)
    parser.add_argument("--skewing_idx_path", type=str, default=None)
    
    parser.add_argument("--lr_proj_path", type=str, default=None)
    
    parser.add_argument("--lr_att_mode", type=str, default='none')
    parser.add_argument("--budget", type=float, default=0.1)
    
    parser.add_argument("--save_name", type=str, default=None)
    parser.add_argument("--start_layer", type=str, default='1')
    parser.add_argument("--token_group", type=int, default=1)
    
    parser.add_argument("--save_video_emb", action='store_true')
    
    args = parser.parse_args()
    
    print(args)
    
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    np.random.seed(1234)
    random.seed(1234)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    if "deepseek" in args.model_path.lower():
        if "qwen3" in args.model_path.lower():
            args.model_type = 'ds_qwen3'
        elif "llama" in args.model_path.lower():
            args.model_type = 'ds_llama'
        else:
            raise ValueError("DeepSeek model type not supported")
    elif 'internvl3' in args.model_path.lower():
        args.model_type = 'internvl3'
    elif 'llama' in args.model_path.lower():
        args.model_type = 'llama'
    elif 'qwen2.5-vl' in args.model_path.lower():
        args.model_type = 'qwen2.5vl'
    elif 'qwen3' in args.model_path.lower():
        args.model_type = 'qwen3'
    else:
        raise ValueError("Model type not supported")
    
    VIDEO_LLMS = ('qwen2.5vl', 'internvl3')
    
    args.model_name = args.model_path.split("/")[-1]
    
    args.eval_mode = 'gen'
    attn_implementation = 'flash_attention_2'
    

    transformers.models.llama.modeling_llama.eager_attention_forward = llama_eager_attention_forward
    transformers.models.llama.modeling_llama.LlamaAttention.forward = llama_att_forward
    transformers.models.llama.modeling_llama.LlamaModel.forward = llama_model_forward

    transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLFlashAttention2.forward = qwen2_5_vl_att_forward
    transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLModel.forward = qwen2_5_vl_model_forward
    
    transformers.models.qwen3.modeling_qwen3.eager_attention_forward = qwen3_eager_attention_forward
    transformers.models.qwen3.modeling_qwen3.Qwen3Attention.forward = qwen3_att_forward
    transformers.models.qwen3.modeling_qwen3.Qwen3Model.forward = qwen3_model_forward

    if args.save_video_emb and args.model_type in VIDEO_LLMS:
        transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLModel = Catcher_Qwen2_5_VLModel

    # transformers.models.opt.modeling_opt.OPTAttention.forward = opt_att_forward
    # transformers.models.opt.modeling_opt.OPTDecoder.forward = opt_decoder_forward
        
    torch_dtype = torch.bfloat16
    args.torch_dtype = torch_dtype
    device_map = 'auto'
    if args.lr_att_mode.startswith('shadowkv'):
        device_map = 'cpu'

    extra_kwargs = {}

    if args.model_type == 'qwen2.5vl':
        from transformers import Qwen2_5_VLForConditionalGeneration
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model_path,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
            device_map=device_map,).eval()
        processor = AutoProcessor.from_pretrained(args.model_path)
        print(f"Model config - Max position embeddings: {model.config.max_position_embeddings}", flush=True)
        print(f"Tokenizer max length: {processor.tokenizer.model_max_length}", flush=True)
    elif args.model_type == 'videollama3':
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            trust_remote_code=True,
            device_map=device_map,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
        ).eval()
        processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    elif args.model_type == 'internvl3':
        from models.internvl3.modeling_internvl_chat import InternVLChatModel
        model = InternVLChatModel.from_pretrained(
            args.model_path,
            torch_dtype=torch_dtype,
            load_in_8bit=False,
            use_flash_attn=attn_implementation=='flash_attention_2',
            device_map=device_map).eval()
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, use_fast=False)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_path, 
                                                    attn_implementation=attn_implementation,
                                                    torch_dtype=torch_dtype, 
                                                    device_map=device_map, 
                                                    **extra_kwargs
                                                    ).eval()
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False) 
    
    from transformers import GenerationConfig    
    try:
        gen_configs = GenerationConfig.from_pretrained(args.model_path, trust_remote_code=False)
    except Exception as e:
        assert args.model_type == 'ds_qwen3', "model_type should be ds_qwen3"
        gen_configs = GenerationConfig(
            temperature=0.6,
            top_p=0.95,
            do_sample=True,
            bos_token_id=151643, 
            eos_token_id=151645
        )
    print("gen_configs=", gen_configs)


    if args.lr_att_mode.startswith('shadowkv'):
        decoder_model = None
        outlier_chunk = None
        local_chunk = None
        try:
            _, chunk_size, rank, outlier_chunk, local_chunk = args.lr_att_mode.split('-')
            outlier_chunk = int(outlier_chunk)
            local_chunk = int(local_chunk)
        except Exception as e:
            _, chunk_size, rank = args.lr_att_mode.split('-')
        
        rank = int(rank)
        chunk_size = int(chunk_size)
        assert args.budget > 1, f"only support sparse budget > 1, got {args.budget}"
        sparse_budget = int(args.budget)
        print(f"Shadowkv rank={rank}, chunk_size={chunk_size}, sparse_budget={sparse_budget}, outlier_chunk={outlier_chunk}, local_chunk={local_chunk}", flush=True)
        from shadowkv.models import choose_model_class
        LLM = choose_model_class(args.model_type)
        model = LLM(model_name=args.model_path, hf_model=model,
                    batch_size=1, max_length=args.seq_len+2048, attn_mode='shadowkv', dtype=torch.bfloat16, 
                    sparse_budget=sparse_budget, rank=rank, chunk_size=chunk_size, minference=False, 
                    outlier_chunk=outlier_chunk, local_chunk=local_chunk)

    else:
        if args.model_type == 'opt':
            decoder_model = model.model.decoder
        elif args.model_type == 'internvl3':
            decoder_model = model.language_model.model
        else:
            decoder_model = model.model

        if '0' in args.start_layer:
            use_emb = False
            try:
                args.start_layer, use_cur_hidden_forl0, use_emb = args.start_layer.split('-')
                use_emb = use_emb == 'emb'
            except Exception as e:
                args.start_layer, use_cur_hidden_forl0 = args.start_layer.split('-')
                
            use_cur_hidden_forl0 = use_cur_hidden_forl0 == 'curr'
            print(f"Using start_layer={args.start_layer}, use_cur_hidden_forl0={use_cur_hidden_forl0}, use_emb={use_emb}", flush=True)

        args.start_layer = int(args.start_layer)
        decoder_model.start_idx = args.start_layer

        if 'infinigen' in args.lr_att_mode:
            if os.path.exists(args.skewing_matrix_path):
                print("Loading skewing matrix from ", args.skewing_matrix_path)
                skewing_matrix = torch.load(args.skewing_matrix_path, map_location='cpu').to(torch_dtype)
                for layer_i in range(len(decoder_model.layers)):
                    decoder_model.layers[layer_i].self_attn.skewing_matrix = skewing_matrix[layer_i].clone()
            else:
                raise ValueError(f"skewing_matrix_path={args.skewing_matrix_path} not found")
            
            if os.path.exists(args.skewing_idx_path+'/0.pt'):
                print("Loading skewing idx from ", args.skewing_idx_path)
                kv_groups = decoder_model.layers[0].self_attn.num_key_value_groups if hasattr(decoder_model.layers[0].self_attn, 'num_key_value_groups') else 1
                head_dim = decoder_model.layers[0].self_attn.head_dim
                for layer_i in range(len(decoder_model.layers)):
                    skewing_idx = torch.load(args.skewing_idx_path + f"/{layer_i}.pt", map_location='cpu') # h, r
                    if layer_i == 0:
                        print("skewing_idx.shape=", skewing_idx.shape)
                    decoder_model.layers[layer_i].self_attn.skewing_idx = skewing_idx
                lowrank_ratio = skewing_idx.shape[1] / head_dim
                print(f"Low rank ratio={lowrank_ratio:.4f}")
            else:
                raise ValueError(f"skewing_idx_path={args.skewing_idx_path} not found")

        elif args.lr_att_mode.startswith('loki'):
            print("Loading lr_proj from ", args.lr_proj_path)
            for layer_i in range(len(decoder_model.layers)):
                loki_proj = torch.load(args.lr_proj_path + '/loki_proj_'+str(layer_i)+'.pt', map_location='cpu').to(torch_dtype)
                assert torch.isfinite(loki_proj).all(), f"loki_proj contains NaN or Inf values in layer {layer_i}"
                decoder_model.layers[layer_i].self_attn.loki_proj = loki_proj
                if layer_i == 0: print("loki_proj.shape=", loki_proj.shape)

            head_dim = decoder_model.layers[0].self_attn.head_dim
            lowrank_ratio = loki_proj.shape[2] / head_dim        
            print(f"Low rank ratio={lowrank_ratio:.4f}")

        elif args.lr_att_mode.startswith('lr_proj'):
            print("Loading lr_proj from ", args.lr_proj_path)
            for layer_i in range(len(decoder_model.layers)):
                lr_kproj = torch.load(args.lr_proj_path + '/lr_kproj_'+str(layer_i)+'.pt', map_location='cpu').to(torch_dtype)
                decoder_model.layers[layer_i].self_attn.lr_kproj = lr_kproj
                if layer_i == 0: print("lr_kproj.shape=", lr_kproj.shape)
            kv_groups = decoder_model.layers[0].self_attn.num_key_value_groups if hasattr(decoder_model.layers[0].self_attn, 'num_key_value_groups') else 1
            head_dim = decoder_model.layers[0].self_attn.head_dim
            lowrank_ratio = lr_kproj.shape[1] / (decoder_model.layers[0].self_attn.config.num_attention_heads / kv_groups) / head_dim        
            print(f"Low rank ratio={lowrank_ratio:.4f}")

        elif args.lr_att_mode == 'none':
            pass
        else:
            raise ValueError(f"args.lr_att_mode={args.lr_att_mode} not supported")

        for layer_i in range(len(decoder_model.layers)):
            decoder_model.layers[layer_i].self_attn.budget = args.budget
            decoder_model.layers[layer_i].self_attn.lr_att_mode = args.lr_att_mode
            if args.start_layer == 0:
                if layer_i == 0:
                    decoder_model.layers[layer_i].self_attn.use_emb = use_emb
                    decoder_model.layers[layer_i].self_attn.use_cur_hidden = True
                elif layer_i == 1:
                    decoder_model.layers[layer_i].self_attn.use_cur_hidden = use_cur_hidden_forl0
            decoder_model.layers[layer_i].self_attn.token_group = args.token_group
            decoder_model.layers[layer_i].self_attn.eval_mode = args.eval_mode
    
    if args.tasks != 'none':
            
        density_list = []
        fetch_num_list = []
        args.tasks = args.tasks.split(",")
        assert len(args.tasks) == 1, f"args.tasks={args.tasks} should be a single task"
        task = args.tasks[0]
        
        out_file_list = []
        
        save_path = os.path.dirname(args.save_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        if 'longbench' in task:
            dataset2prompt = json.load(open("./bench/LongBench/LongBench-v1/config/dataset2prompt.json", "r"))
            dataset2maxlen = json.load(open("./bench/LongBench/LongBench-v1/config/dataset2maxlen.json", "r"))
    
            if '_mqa+sum' in task:
                datasets = ["hotpotqa", "2wikimqa", "musique", "gov_report", "multi_news", "qmsum"]  
            elif '_mqa' in task:
                datasets = ["hotpotqa", "2wikimqa", "musique"]  
            else:
                datasets = ["narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique", \
                            "gov_report", "qmsum", "multi_news", "trec", "triviaqa", "samsum", \
                            "passage_retrieval_en", "lcc", "repobench-p"]                        
            datasets_for_cot = ["hotpotqa", "2wikimqa", "musique"]                        
            
            if 'cot_only' in task:
                datasets = datasets_for_cot
            elif 'cot' in task:
                assert False, 'evaluate cot and normal separately'

            for dataset in datasets:
                out_file_list = []
                dataset_name = dataset
                filename = f'./bench/LongBench/LongBench-v1/{dataset_name}.jsonl'                
                data_all = [json.loads(line) for line in open(filename, encoding='utf-8')]
                print(f"Loaded {len(data_all)} samples from {filename}")
                
                if "cot" in task:
                    assert args.model_type in ['ds_qwen3', 'qwen3', 'ds_llama'], f"Model type {args.model_type} not supported for cot"
                    if not 'cot_only' in task:
                        out_file_list.append(args.save_name + f"_{dataset_name}.jsonl")
                    out_file_list.append(args.save_name + f"_{dataset_name}_cot.jsonl") 
                else:
                    out_file_list.append(args.save_name + f"_{dataset_name}.jsonl")                 
                          
                prompt_format = dataset2prompt[dataset]
                max_gen = dataset2maxlen[dataset]        
                process_out_file_list(args, model, decoder_model, out_file_list, data_all, 
                                      task, gen_configs, tokenizer, prompt_format=prompt_format, max_gen=max_gen)

        elif 'needle' in task:
            prompt_dirs = os.listdir('./bench/Needle_test/prompts')
            prompt_dir = None
            for prompt_dir_ in prompt_dirs:
                if args.model_path.split("/")[-1] in prompt_dir_ and str(args.seq_len) in prompt_dir_:
                    prompt_dir = os.path.join('./bench/Needle_test/prompts', prompt_dir_)
                    break
            if prompt_dir is None:
                print(f"Cannot find prompt dir for {args.model_path} and seq_len {args.seq_len}")
                raise ValueError("Please check the prompt directory for Needle task")
            print(f"Using prompt dir {prompt_dir}")
            out_file_list = []
            data_all = []
            for filename in glob.glob(f'{prompt_dir}/*_prompts.json'):
                with open(filename, 'r') as f:
                    prompts = json.load(f)            
                id = os.path.basename(filename).split("_prompts.json")[0]
                data_all.append({"_id": id, 'content': prompts})
            print(f"Loaded {len(data_all)} samples from {prompt_dir}")
            if "cot" in task:
                assert False, "Not used for needle task"
            else:
                out_file_list.append(args.save_name + ".jsonl")
            process_out_file_list(args, model, decoder_model, out_file_list, data_all, task, gen_configs, tokenizer)
        elif 'ruler' in task:
            from datasets import load_dataset
            if args.model_type == 'llama':
                model_dir = 'Llama'
            elif args.model_type == 'qwen3':
                model_dir = 'Qwen3'
            else:
                raise Exception("Model not found", args.model_type)
            
            RULER_DATA_DIR = './bench/RULER/DATA'

            if "_sub" in task:
                ruler_tasks = ['niah_multivalue', 'vt', 'qa_1']
            else:
                ruler_tasks = ['niah_single_1', 'niah_single_2', 'niah_multikey_1', 
                               'niah_multiquery', 'niah_multivalue', 'vt', 'qa_1', 'qa_2']
                
            if "cot" in task:
                assert False, "Not used for ruler task"
            
            for ruler_task in ruler_tasks:
                dataset = load_dataset("json", data_files=f'{RULER_DATA_DIR}/{model_dir}/synthetic/{args.seq_len}/data/{ruler_task}/validation.jsonl', split='train')
                out_file_list = []
                data_all = []
                print(f"{ruler_task} task: {len(dataset)} samples loaded")
                for i in range(len(dataset)):
                    input_text = dataset[i]['input']
                    answer_prefix = dataset[i]['answer_prefix'] if 'answer_prefix' in dataset[i] else ''
                    id = str(i)
                    gt = dataset[i]['outputs']
                    data_all.append({"_id": id, 'content': input_text, 'gt': gt, 'answer_prefix': answer_prefix})
                out_file_list.append(args.save_name + f"_{ruler_task}.jsonl")
                if 'niah' in ruler_task:
                    max_gen = 128
                elif 'vt' in ruler_task:
                    max_gen = 30
                elif 'cwe' in ruler_task:
                    max_gen = 120
                elif 'fwe' in ruler_task:
                    max_gen = 50
                elif 'qa' in ruler_task:
                    max_gen = 32
                else:
                    raise Exception("Gen len not found")
                process_out_file_list(args, model, decoder_model, out_file_list, data_all, task, gen_configs, tokenizer, max_gen=max_gen)
                
        elif 'mlvu' in task:

            args.eval_samples = int(args.eval_samples)
            assert args.eval_samples == 0 or args.eval_samples > 1, f"args.eval_samples={args.eval_samples} should be 0 or > 1"

            assert args.model_type in VIDEO_LLMS, f"Model type {args.model_type} not supported for MLVU task"
            if '_cot' in task:
                assert False, "Not used for MLVU task"
            else:
                do_cot = 0
            MLVU_DATA_DIR = './bench/MLVU/MVLU_DATA/MLVU'
            if args.model_type == 'qwen2.5vl':
                PROCESSED_MLVU_DATA_DIR = f'./bench/MLVU/MVLU_DATA/Qwen2.5-VL_processed'
            elif args.model_type == 'videollama3':
                PROCESSED_MLVU_DATA_DIR = f'./bench/MLVU/MVLU_DATA/VideoLlama3_processed'
            elif args.model_type == 'internvl3':
                PROCESSED_MLVU_DATA_DIR = f'./bench/MLVU/MVLU_DATA/InternVL3_processed'
            else:
                raise ValueError(f"Model type {args.model_type} not supported for MLVU task")
            DO_REUSE_PROCESSED = True
            DO_SAVE_PROCESSED = True
            if not os.path.exists(PROCESSED_MLVU_DATA_DIR):   
                os.makedirs(PROCESSED_MLVU_DATA_DIR)
            from dataset import MLVU
            data_list = {
                "subPlot": ("8_sub_scene.json", f"{MLVU_DATA_DIR}/video/8_sub_scene", "video", False),
                "summary": ("9_summary.json", f"{MLVU_DATA_DIR}/video/9_summary", "video", False)
            }
            dataset = MLVU(f"{MLVU_DATA_DIR}/json", data_list, max_samples=args.eval_samples)

            subplot_out_name = args.save_name +  "_subplot_all.json"
            summary_out_name = args.save_name +  "_summary_all.json"
            
            subplot_has_data = {}
            if os.path.exists(subplot_out_name):
                subplot_has_data = {json.loads(line)["video_name"]:0 for line in open(subplot_out_name, "r")}
            summary_has_data = {}
            if os.path.exists(summary_out_name):
                summary_has_data = {json.loads(line)["video_name"]:0 for line in open(summary_out_name, "r")}

            subplot_f = open(subplot_out_name, "a")
            summary_f = open(summary_out_name, "a")

            clear_layer_stats(decoder_model)

            for example in tqdm(dataset):
                video_path = example["video"]
                video_name = video_path.split("/")[-1]
                task_type = example["task_type"]
                if task_type == "subPlot":
                    if video_name in subplot_has_data:
                        print(f"Skip {video_name} for subplot", flush=True)
                        continue
                elif task_type == "summary":
                    if video_name in summary_has_data:
                        print(f"Skip {video_name} for summary", flush=True)
                        continue
                else:
                    raise ValueError(f"Unknown task type: {task_type}")
                
                max_new_tokens = 500 if do_cot==0 else 5000
                INSTRUCT_COT = "Carefully watch this video and pay attention to every detail. Based on your observations, answer the given questions. Please think step by step and provide a detailed reasoning process before giving your final answer."
                INSTRUCT = "Carefully watch this video and pay attention to every detail. Based on your observations, answer the given questions."
                
                sys_ins = INSTRUCT if do_cot==0 else INSTRUCT_COT
                # print(f"Processing {video_path} for {task_type} task", flush=True)
                
                if args.model_type == 'internvl3':
                    output, inputs_meta, response_len = inference_single_video_internvl3(args, video_path, '', 
                                                                sys_ins + '\n\n' + example["question"], model, tokenizer, 
                                                                max_new_tokens, gen_configs, PROCESSED_MLVU_DATA_DIR, DO_REUSE_PROCESSED, DO_SAVE_PROCESSED)
                else:
                    output, inputs_meta, response_len = inference_single_video(args, video_path, '', 
                                                                sys_ins + '\n\n' + example["question"], model, processor, 
                                                                max_new_tokens, gen_configs, PROCESSED_MLVU_DATA_DIR, DO_REUSE_PROCESSED, DO_SAVE_PROCESSED)
                result = {}
                result["video_name"] = video_name
                result['Q'] = example["question"]
                result['A'] = example['answer']
                result['pred'] = output.strip()
                result['response_len'] = response_len
                result['video_path'] = video_path
                result['duration'] = example['duration']
                result['inputs_meta'] = inputs_meta
                
                if task_type=="subPlot":
                    subplot_f.write(json.dumps(result) + "\n")
                    subplot_f.flush()
                elif task_type=="summary":
                    summary_f.write(json.dumps(result) + "\n")
                    summary_f.flush()
                else:
                    raise ValueError(f"Unknown task type: {task_type}")


            subplot_f.close()
            summary_f.close()
            print("Evaluation completed.", flush=True)      
            print_layer_stats(decoder_model, model)

        else:
            NotImplementedError(f"task {task} not implemented")
            
