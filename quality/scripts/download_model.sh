echo "Downloading models..."
mkdir -p MODELS

mode=$1


download_model() {
# 输入提示问用户你是否以及有了Llama-3.1-8B-Instruct, Qwen3-4B, Qwen3-8B, Qwen2.5-VL-3B-Instruct, Qwen2.5-VL-7B-Instruct的huggingface权重文件
# 如果有了则需要将这些模型文件夹链接到MODELS目录下，例如MODELS/Llama-3.1-8B-Instruct
# 如果没有则下面开始下载这些模型文件

}

download_model2() {
# 输入提示问用户你是否以及有了Qwen3-14B和InternVL3-14B的huggingface权重文件
# 如果有了则需要将这些模型文件夹链接到MODELS目录下，例如MODELS/Qwen3-14B
# 如果没有则下面开始下载这些模型文件

}

if [ "$mode" == "full" ]; then

    

else



fi



# git clone https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct MODELS/Llama-3.1-8B-Instruct
# git clone https://huggingface.co/Qwen/Qwen3-4B MODELS/Qwen3-4B
# git clone https://huggingface.co/Qwen/Qwen3-8B MODELS/Qwen3-8B
# git clone https://huggingface.co/Qwen/Qwen3-14B MODELS/Qwen3-14B

# git clone https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct MODELS/Qwen2.5-VL-3B-Instruct
# git clone https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct MODELS/Qwen2.5-VL-7B-Instruct
# git clone https://huggingface.co/OpenGVLab/InternVL3-14B MODELS/InternVL3-14B

echo "--------------------------------"