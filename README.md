# SLoMO-AD-Competition-2026

Solution for SLoMO-AD Competition

## Solution Description

For each video in the dataset, the audio track is extracted. Then, the voice is separated from the rest of the audio using the [MSST](https://github.com/ZFTurbo/Music-Source-Separation-Training/) repository and the [BS Roformer](https://huggingface.co/noblebarkrr/mvsepless_resources/tree/main/bs_roformer) model. Next, the vocals are transcribed with timestamps using the [parakeet_v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) model. 

After that, we work with the video itself, the resulting text, and the set of characters. For each time interval for which we need to generate a comment, the corresponding segment is extracted from the video. Additionally, the following are provided along with the video segment:
1) Images and names of all characters in this clip. 
2) A text transcript covering this segment (if available).
3) A limit is set on the number of words in the response, calculated at a maximum of 3 words per second of video.

The following prompt is generated for the model:

```
You are given {number of actors} reference images of characters, one main video, and a transcript of the dialogue occurring in and around this video clip.
--- CHARACTER REFERENCES ---
{actor_references}
--- DIALOGUE CONTEXT ---
{dialogues_text}
--- INSTRUCTIONS ---
Please provide a detailed visual description of the main actions, subjects, and scene changes IN THE VIDEO.
Use the dialogue context only to better understand the situation, but describe only what is visibly happening.
When describing people, if you visually recognize any of the referenced characters, use their exact names.
Crucially, if a referenced character is NOT visible in the video, do NOT mention them at all (even if they are speaking in the dialogue).
CRITICAL REQUIREMENT: Your entire response MUST be strictly under {words_count} words!
```

[Qwen3.8-27B-FP8](https://huggingface.co/Qwen/Qwen3.8-27B-FP8) in Reasoning mode was used as the primary model for the Main Track. The [Qwen3-VL-32B-Instruct-FP8](https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct-FP8) model also performed quite well. It yielded a slightly lower metric, but demonstrated a significantly higher processing speed. 

For the Special Track, the [Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) model with 8 billion parameters was used. Surprisingly, despite its small size, it produces rather good results.

*Note: For this solution, the models were not fine-tuned and were used out-of-the-box.*

## Requirements

All runs were performed on a single NVIDIA A6000 Blackwell 96GB GPU.

## Dataset

Dataset consists of two parts ([CMD-AD](https://huggingface.co/datasets/Jyxarthur/CMD-AD) and [SF20K](https://huggingface.co/datasets/Jyxarthur/sf20kad-private-set/tree/main))

| Split (no. movies) | 	Train	| Public test	| Private test                                                                                                      |
| : ---- | : ----- | :------ |:------------------------------------------------------------------------------------------------------------------|
| CMD-AD	| 1,332	| 98	| 44 ([Videos](https://huggingface.co/datasets/Jyxarthur/CMD-AD/resolve/main/video_private_test.zip?download=true)) | 
| SF20K (zero-shot)	| –	| 17	| 26 ([Videos](https://huggingface.co/datasets/Jyxarthur/sf20kad-private-set/resolve/main/sf20kad_private_set_video.zip?download=true))                                                                                                   |

Code is prepared to run on Private Test data.

### CMD-AD

Download CMD-AD private set data to folder `./input/CMD-AD/`
* video_private_test.zip
* actor_profiles_private_test.tar
* cmdad_ad_private_test.csv
* cmdad_charbank_private_test.json

Unzip and untar archives in folders `video_private_test` and `actor_profiles_private_test`.

### SF20K-AD

Download SF20K-AD private set data to folder `./input/SF20K-AD/`
* sf20kad_private_set_video.zip
* char_bank_private_set.tar
* sf20kad_private_set.csv
* sf20kad_private_set_charbank.json

Unzip and untar archives in folders `sf20kad_private_set_video` and `char_bank_private_set`.

## Pipeline

Place videos and all additional data in the `./input/` folder.

### Preprocessing

```bash
python preproc_data.py ./input/
```

### Run Inference for the Main Track

First, start the vLLM server in your first terminal:
```bash
CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_ORDER=PCI_BUS_ID \
VLLM_USE_FLASHINFER_SAMPLER=0 \
VLLM_USE_DEEP_GEMM=0 VLLM_USE_DEEP_GEMM_E8M0=0 \
python -m vllm.entrypoints.openai.api_server \
--model Qwen/Qwen3.8-27B-FP8 \
--served-model-name qwen38-27bfp8 \
--tensor-parallel-size 1 \
--max-num-seqs 64 \
--gpu-memory-utilization 0.95 \
--host localhost \
--port 8000 \
--dtype auto \
--max-model-len 131072 \
--limit-mm-per-prompt.video 1 \
--limit-mm-per-prompt.image 300 \
--allowed-local-media-path {your_local_path_with_dataset} \
--mm-processor-kwargs '{"fps": 2}' \
--media-io-kwargs '{"video": {"num_frames": -1}}'
```

* **Note:** You must provide the correct dataset path via the `--allowed-local-media-path` argument, otherwise the server won't be able to access the media files.

Then, run the main processing script in a second terminal:
```bash
python run_inference_qwen38_27B_cmd_ad.py ./input/
python run_inference_qwen38_27B_sf20k.py ./input/
```

### Run Inference for the Special Track

First, start the vLLM server in your first terminal:
```bash
CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_ORDER=PCI_BUS_ID \
VLLM_USE_FLASHINFER_SAMPLER=0 \
VLLM_USE_DEEP_GEMM=0 VLLM_USE_DEEP_GEMM_E8M0=0 \
python -m vllm.entrypoints.openai.api_server \
--model Qwen/Qwen3-VL-8B-Instruct \
--served-model-name qwen3-8b \
--tensor-parallel-size 1 \
--max-num-seqs 32 \
--gpu-memory-utilization 0.95 \
--host localhost \
--port 8000 \
--dtype auto \
--max-model-len 131072 \
--allowed-local-media-path {your_local_path_with_dataset} \
--mm-processor-kwargs '{"fps": 2}' \
--media-io-kwargs '{"video": {"num_frames": -1}}'
```

* **Note:** You must provide the correct dataset path via the `--allowed-local-media-path` argument, otherwise the server won't be able to access the media files.

Then, run the special track processing script in a second terminal:
```bash
python run_inference_qwen3_vl_8B_cmd_ad.py ./input/
python run_inference_qwen3_vl_8B_sf20k.py ./input/
```

# Results

Results on public test leaderboard:

| Model                       | Track   | LB Results (AD Score) |
|:----------------------------|:--------|:----------------------|
| [Qwen3.8-27B-FP8](https://huggingface.co/Qwen/Qwen3.8-27B-FP8) (Reasoning) | Main    | 50.07                 |
| [Qwen3-VL-32B-Instruct-FP8](https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct-FP8)   | Main    | ---                   |
| [Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)   | Special | 46.83                 |

For reference TOP-1 solution: AD Score: 54.49.

