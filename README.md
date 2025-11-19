# HW3
雲端連結: https://drive.google.com/drive/folders/1wKoqzkjlOxnEWRG9oNuAERdfvOjf9Pg0?usp=sharing
雲端連結中有Task1和Task2的結果以及best_model幫助重現

Conda 環境:
為了方便使用，我們提供了一個 Conda 環境配置文件。使用以下命令來創建並激活環境：
conda env create -f hw3_environment.yml
conda activate hw3

***GPT2***
Task1：
task1/train_GPT2.py：用於訓練基於 GPT-2 的音樂生成模型。使用 MIDI 文件作為訓練數據，並使用字典文件將音樂事件映射到詞語。
task1/test_unconditional.py：測試訓練好的模型，根據給定的條件生成音樂。

要重現Task1的生成結果可使用: python task1/test_unconditional.py --model_path ./trained_model/best_model.pkl --prompt_path ./prompt_midi.mid --output_path ./generated_midi.mid --n_target_bar 24 --temperature 1.2 --topk 5


Task2:
task2/task2.py：使用訓練好的模型，根據提供的提示生成音樂延續部分。支持多次生成和參數調整。

運行以下指令生成音樂延續： python task2/task2.py --model_path ./trained_model/best_model.pkl --prompt_path ./prompt_midi.mid --output_path ./generated_midi.mid --n_target_bar 24 --temperature 1.2 --topk 5 --dict_path ./dictionary.pkl
************************************************************************************************************************************************************************************************************************************************************

***Transformer***
Task1: 
Transformer/Task1/train.py：用於訓練基於Transformer的音樂生成模型。
Transformer/Task1/test_unconditional.py：測試訓練好的模型，根據給定的條件生成音樂。
要重現Task1的生成結果可使用:
python test_unconditional.py \
    --dict_path ./dictionary.pkl \
    --model_path ./model_checkpoint.pkl \
    --prompt_path ./prompt_midi.mid \
    --output_path ./generated_music.mid \
    --n_target_bar 32 \
    --temperature 1.2 \
    --topk 5 \
    --n_samples 20
    
Task2:
Transformer/Task2/task2.py：使用訓練好的模型，根據提供的提示生成音樂延續部分。支持多次生成和參數調整。

運行以下指令生成音樂延續：
python generate_continuation.py \
    --dict_path ./dictionary.pkl \
    --model_path ./model_checkpoint.pkl \
    --prompt_path ./prompt_midi.mid \
    --output_path ./generated_music.mid \
    --n_target_bar 24 \
    --temperature 1.2 \
    --topk 5 \
    --n_samples 1


