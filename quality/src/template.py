
def apply_template(args, prompt, tokenizer, enable_cot):
    sys_prompt = "You are a helpful AI bot that answers questions for a user. Keep your response short and direct."
    # sys_prompt = "You are a helpful assistant that answers questions for a user."
    if args.model_type == 'qwen3':
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_cot # Switches between thinking and non-thinking modes. Default is True.
        )
        return text
    elif args.model_type == 'ds_qwen3':
        assert enable_cot is True
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )  
        return text 
    elif args.model_type == 'ds_llama':
        assert enable_cot is True
        # Avoid adding a system prompt; all instructions should be contained within the user prompt.
        messages = [
            {"role": "user", "content": sys_prompt + '\n\n' + prompt},
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )  
        return text 
    elif args.model_type == 'llama':
        assert enable_cot is False
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt},
        ]
        text = tokenizer.apply_chat_template(
            messages, 
            tokenize=False,
            add_generation_prompt=True)
        return text
    elif args.model_type == 'gemma3':
        assert enable_cot is False
        messages = [
            {"role": "system", "content": [{"type": "text", "text": sys_prompt},]},
            {"role": "user", "content": [{"type": "text", "text": prompt},]}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return text
    else:
        raise NotImplementedError(f"Model type {args.model_type} not supported")
    