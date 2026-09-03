# https://eval.ai/web/challenges/challenge-page/2713/overview

import os
import sys
import json
import warnings
import imageio
import cv2
import numpy as np
import pandas as pd
from decord import VideoReader, cpu
from openai import OpenAI

warnings.filterwarnings("ignore")

# INPUT_PATH = '../../input/'
INPUT_PATH = sys.argv[1] + '/'


def read_video_from_path(path):
    try:
        reader = imageio.get_reader(path)
    except Exception as e:
        print("Error opening video file: ", e)
        return None
    frames = []
    for i, im in enumerate(reader):
        frames.append(np.array(im))
    return np.stack(frames)


def proc_with_qwen():
    input_file = INPUT_PATH + '/CMD-AD/cmdad_ad_private_test.csv'

    client = OpenAI(
        base_url="http://127.0.0.1:8000/v1",
        api_key="EMPTY"
    )
    model_name = "qwen38-27bfp8"

    valid = pd.read_csv(input_file)
    valid['video_id'] = valid['cmd_filename']
    videos_to_proc = valid['video_id'].unique()
    print(len(videos_to_proc))

    character_map = json.load(open(INPUT_PATH + '/CMD-AD/cmdad_charbank_private_test.json', 'r'))

    N_FRAMES = 50
    out_file = './' + os.path.basename(input_file)[:-4] + '_private_results_{}_{}_char.csv'.format(N_FRAMES, model_name)
    os.makedirs("./cache_char_vllm/", exist_ok=True)

    results = []
    for video_id in videos_to_proc:
        print(video_id)
        video_id_small = video_id.split("/")[-1]
        video_path = os.path.abspath(INPUT_PATH + '/CMD-AD/video_private_test/' + video_id + '.mp4')
        part = valid[valid['video_id'] == video_id]

        imdbid = part['imdbid'].unique()
        if len(imdbid) > 1:
            print('Strange: ', imdbid, len(imdbid))
        imdbid = imdbid[0]
        chars = character_map[imdbid]
        print(chars)
        print("Characters in movie: {}".format(len(chars)))

        actors_data = []
        for c in chars:
            im_path = os.path.abspath(INPUT_PATH + '/CMD-AD/actor_profiles_private_test/' + c['id'] + '.jpg')
            if not os.path.exists(im_path):
                continue
            actors_data.append({"image": im_path, "name": c['role']})
            if len(actors_data) > 25:
                break

        print("Actors data: {}".format(len(actors_data)))

        df = pd.read_csv(INPUT_PATH + '/CMD-AD/audios_vocals/' + video_id_small + '/vocals_mono_text_segment.csv')
        df = df.values
        txt_arr = []
        for i in range(len(df)):
            txt_arr.append((df[i, 0], df[i, 1], df[i, 2]))
        print(txt_arr)

        actor_references = ""
        for i, actor in enumerate(actors_data):
            actor_references += f"Image {i + 1} is the character '{actor['name']}'. "

        cache_path = "./cache_char_vllm/" + video_id_small + '{}_{}_with_char_and_text.csv'.format(N_FRAMES, model_name)
        if os.path.isfile(cache_path):
            print("Reading from cache: {}".format(cache_path))
            single_file = pd.read_csv(cache_path)
            results.append(single_file)
            continue

        if part.empty:
            continue

        vr = VideoReader(video_path, ctx=cpu(0))
        total_frames = len(vr)
        original_fps = vr.get_avg_fps()
        video_duration = total_frames / original_fps

        print(f"Frames in file: {total_frames}")
        print(f"Length of video: {video_duration:.2f} sec, FPS: {original_fps:.2f}")

        single_results = {}
        for index, row in part.iterrows():
            ad_id = row['ad_id']
            start_index = max(row['start'], 0)
            end_index = row['end']
            duration = end_index - start_index
            start_frame = int(start_index * original_fps)
            end_frame = int(end_index * original_fps)
            print(f"Start {start_index} end {end_index} length: {end_index - start_index} sec")

            part_text = []
            for i in range(len(txt_arr)):
                start1, end1, text1 = txt_arr[i]
                if (start_index <= start1 <= end_index) or (start_index <= end1 <= end_index):
                    part_text.append(txt_arr[i])
                elif (end_index > start1 - 3 and start_index < start1):
                    part_text.append(txt_arr[i])
                elif (start_index < end1 + 3 and end_index > end1):
                    part_text.append(txt_arr[i])
            print(part_text)

            dialogues_text = ''
            for s1, e1, t1 in part_text:
                dialogues_text += "- " + t1 + '\n'
            print(dialogues_text)

            frames = vr[start_frame:end_frame + 1]
            frames_np = frames.asnumpy()
            print(f"Frames extracted: {frames_np.shape[0]}")
            print(f"Array shape: {frames_np.shape}")  # (51, H, W, 3)

            tmp_file = os.path.abspath(f'cache_char_vllm/{video_id_small}.{start_frame}_tmp.mp4')

            height, width, _ = frames_np[0].shape

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(tmp_file, fourcc, original_fps, (width, height))

            for frame in frames_np:
                bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(bgr_frame)

            out.release()
            print(f"Video saved in {tmp_file}")

            target_fps = N_FRAMES / duration
            target_fps = min(target_fps, original_fps)
            words_count = max(int(duration * 3.0), 2)
            print('Target FPS: {} Required words: {}'.format(target_fps, words_count))

            if len(part_text) > 0:
                instruction_text = (
                    f"You are given {len(actors_data)} reference images of characters, one main video, and a transcript of the dialogue occurring in and around this video clip.\n\n"
                    f"--- CHARACTER REFERENCES ---\n"
                    f"{actor_references}\n\n"
                    f"--- DIALOGUE CONTEXT ---\n"
                    f"{dialogues_text}\n\n"
                    f"--- INSTRUCTIONS ---\n"
                    "Please provide a detailed visual description of the main actions, subjects, and scene changes IN THE VIDEO. "
                    "Use the dialogue context only to better understand the situation, but describe only what is visibly happening. "
                    "When describing people, if you visually recognize any of the referenced characters, use their exact names. "
                    "Crucially, if a referenced character is NOT visible in the video, do NOT mention them at all (even if they are speaking in the dialogue). "
                    f"CRITICAL REQUIREMENT: Your entire response MUST be strictly under {words_count} words!"
                )
            else:
                instruction_text = (
                    f"You are given {len(actors_data)} reference images of characters and one main video. "
                    f"{actor_references} "
                    "Please provide a detailed description of the main actions, subjects, and scene changes in the video. "
                    "When describing people, if you recognize any of the referenced characters, use their exact names. "
                    "Crucially, if a referenced character is NOT in the video, do NOT mention them at all (do not state that they are missing). "
                    f"CRITICAL REQUIREMENT: Your entire response MUST be strictly under {words_count} words!"
                )

            prompt_multimodal = []

            for actor in actors_data:
                img_path = actor["image"]
                if not img_path.startswith("file://"):
                    img_path = f"file://{img_path}"

                prompt_multimodal.append({
                    "type": "image_url",
                    "image_url": {
                        "url": img_path
                    }
                })

            vid_path = tmp_file
            if not vid_path.startswith("file://"):
                vid_path = f"file://{vid_path}"

            prompt_multimodal.append({
                "type": "video_url",
                "video_url": {
                    "url": vid_path
                }
            })

            prompt_multimodal.append({
                "type": "text",
                "text": instruction_text
            })

            messages = [
                {
                    "role": "user",
                    "content": prompt_multimodal,
                }
            ]

            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=1.0,
                    top_p=0.95,
                    presence_penalty=0.0,
                    max_tokens=8 * 4096,
                    extra_body={
                        "mm_processor_kwargs": {
                            "fps": target_fps,
                            "min_pixels": 4 * 32 * 32,
                            "max_pixels": 360 * 420
                        },
                        "chat_template_kwargs": {
                            "enable_thinking": True,  # on by default
                            "preserve_thinking": True,  # on by default
                        },
                    },
                    reasoning_effort="xhigh",
                )

                output_text = response.choices[0].message.content.strip()

                debug_response = str(messages) + "\n\n-------- OUTPUT -------" + output_text
                cache_path_txt = ("./cache_char_vllm/" + video_id_small +
                                  '_{}_{}_{}_with_char_and_text.csv'.format(ad_id.replace('/', '_'), N_FRAMES, model_name))
                out = open(cache_path_txt, 'w', encoding='utf-8')
                out.write(debug_response)
                out.close()

            except Exception as e:
                print(f"Error for question {ad_id} for video {video_id}: {e}")
                output_text = ""

            if "</think>" in output_text:
                output_text = output_text.split("</think>")[-1].strip()
            else:
                output_text = output_text

            single_results[ad_id] = output_text
            print("A:", output_text)
            os.remove(tmp_file)

        part['text'] = part['ad_id'].map(single_results)
        part.to_csv(cache_path, index=False)
        results.append(part)

    results = pd.concat(results, axis=0)
    results.to_csv(out_file, index=False)
    print("Results were written to: ", out_file)


if __name__ == "__main__":
    proc_with_qwen()