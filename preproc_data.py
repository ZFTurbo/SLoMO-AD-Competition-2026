import os

if __name__ == '__main__':
    gpu_use = "0"
    print('GPU use: {}'.format(gpu_use))
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_use)
    code_path = os.path.dirname(os.path.abspath(__file__)) + '/'
    os.environ['HF_HOME'] = code_path + 'local_models_cache/'

import glob
import warnings
import sys
import numpy as np
import pandas as pd
import librosa
import tempfile
import soundfile as sf
import urllib.request
import nemo.collections.asr as nemo_asr

warnings.filterwarnings("ignore")
import subprocess


# INPUT_PATH = '../../input/'
INPUT_PATH = sys.argv[1] + '/'


def extract_audio(video_path, audio_path):
    command = [
        "ffmpeg",
        "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "44100",
        "-ac", "2",
        "-y",
        audio_path
    ]

    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def extract_files_1():
    out_audio_folder = INPUT_PATH + '/SF20K-AD/audios/'
    os.makedirs(out_audio_folder, exist_ok=True)
    files = glob.glob(INPUT_PATH + '/SF20K-AD/sf20kad_private_set_video/*.*')
    print(len(files))

    for f in files:
        print(f)
        out_audio_path = out_audio_folder + os.path.basename(f)[:-4] + '.wav'
        extract_audio(f, out_audio_path)


def extract_files_2():
    out_audio_folder = INPUT_PATH + '/CMD-AD/audios/'
    os.makedirs(out_audio_folder, exist_ok=True)
    files = glob.glob(INPUT_PATH + '/CMD-AD/video_private_test/*/*.*')
    print(len(files))

    for f in files:
        print(f)
        out_audio_path = out_audio_folder + os.path.basename(f)[:-4] + '.wav'
        extract_audio(f, out_audio_path)


def extract_vocals_from_audio(type=0):
    if type == 0:
        in_audio_folder = INPUT_PATH + '/SF20K-AD/audios/'
        out_audio_folder = INPUT_PATH + '/SF20K-AD/audios_vocals/'
    else:
        in_audio_folder = INPUT_PATH + '/CMD-AD/audios/'
        out_audio_folder = INPUT_PATH + '/CMD-AD/audios_vocals/'

    os.makedirs(out_audio_folder, exist_ok=True)
    files = glob.glob(in_audio_folder + '*.*')
    print("Extract vocals:", len(files))

    # https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/bs_roformer/bs_6stem_fixed.ckpt?download=true
    # https://huggingface.co/noblebarkrr/mvsepless_resources/raw/main/bs_roformer/bs_6stem_fixed_config.yaml
    check_point_path = code_path + "msst/bs_6stem_fixed.ckpt"
    if not os.path.isfile(check_point_path):
        url = "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/bs_roformer/bs_6stem_fixed.ckpt?download=true"
        print("Downloading BS Roformer checkpoint...")
        os.makedirs(os.path.dirname(check_point_path), exist_ok=True)
        urllib.request.urlretrieve(url, check_point_path)
        print("Download complete!")

    args = {
        'model_type': 'bs_roformer',
        "config_path": code_path + "msst/bs_6stem_fixed_config.yaml",
        "start_check_point": check_point_path,
        "store_dir": out_audio_folder,
        "input_folder": in_audio_folder,
        "device_ids": "0",
    }

    from msst.inference import proc_folder
    proc_folder(args)


def convert_to_mono16k(input_wav, output_wav):
    audio, sr = librosa.load(input_wav, sr=16000, mono=True)
    sf.write(output_wav, audio, sr)


def convert_to_mono(type=0):
    if type == 0:
        in_audio_folder = INPUT_PATH + '/SF20K-AD/audios_vocals/'
    else:
        in_audio_folder = INPUT_PATH + '/CMD-AD/audios_vocals/'
    files = glob.glob(in_audio_folder + '*/vocals.wav')
    print(len(files))
    for f in files:
        convert_to_mono16k(f, f[:-4] + '_mono.wav')


def transcribe_long_audio(asr_model, audio_path, chunk_duration_sec=30, batch_size=4):
    audio, sr = sf.read(audio_path)

    chunk_length_samples = int(chunk_duration_sec * sr)
    total_samples = len(audio)

    all_texts = []
    all_word_timestamps = []
    all_segment_timestamps = []

    with tempfile.TemporaryDirectory() as tmpdir:
        chunk_paths = []

        for i in range(0, total_samples, chunk_length_samples):
            chunk = audio[i: i + chunk_length_samples]
            chunk_path = os.path.join(tmpdir, f"chunk_{i}.wav")
            sf.write(chunk_path, chunk, sr)
            chunk_paths.append(chunk_path)

        print(f"Audio split in {len(chunk_paths)} chunks. Start transcribing...")

        hypotheses = asr_model.transcribe(chunk_paths, batch_size=batch_size, timestamps=True)

        for chunk_idx, hyp in enumerate(hypotheses):
            time_offset = chunk_idx * chunk_duration_sec

            if hyp.text:
                all_texts.append(hyp.text.strip())

            if not hyp.timestamp:
                continue

            if 'word' in hyp.timestamp:
                for stamp in hyp.timestamp['word']:
                    shifted_word = stamp.copy()
                    shifted_word['start'] += time_offset
                    shifted_word['end'] += time_offset
                    all_word_timestamps.append(shifted_word)

            if 'segment' in hyp.timestamp:
                for stamp in hyp.timestamp['segment']:
                    shifted_segment = stamp.copy()
                    shifted_segment['start'] += time_offset
                    shifted_segment['end'] += time_offset
                    all_segment_timestamps.append(shifted_segment)

    full_text = " ".join(all_texts)

    return {
        'text': full_text,
        'words': all_word_timestamps,
        'segments': all_segment_timestamps
    }


def extract_text_from_vocals(type=0):
    asr_model = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v3")

    if type == 0:
        in_audio_folder = INPUT_PATH + '/SF20K-AD/audios_vocals/'
    else:
        in_audio_folder = INPUT_PATH + '/CMD-AD/audios_vocals/'

    files = glob.glob(in_audio_folder + '*/vocals_mono.wav')
    print("Transcribe text from vocals:", len(files))

    for f in files:
        print(f)
        hypotheses = transcribe_long_audio(asr_model, f, chunk_duration_sec=300, batch_size=1)

        out = open(f[:-4] + '_text.txt', 'w', encoding='utf8')
        out.write(hypotheses['text'])
        out.close()

        out = open(f[:-4] + '_text_words.txt', 'w', encoding='utf8')
        word_timestamps = hypotheses['words']
        for stamp in word_timestamps:
            out.write(f"[{stamp['start']:.2f}s - {stamp['end']:.2f}s] : {stamp['word']}\n")
        out.close()

        out = open(f[:-4] + '_text_segment.txt', 'w', encoding='utf8')
        word_timestamps = hypotheses['segments']
        for stamp in word_timestamps:
            out.write(f"[{stamp['start']:.2f}s - {stamp['end']:.2f}s] : {stamp['segment']}\n")
        out.close()

        out_file = f[:-4] + '_text_segment.csv'
        res = []
        word_timestamps = hypotheses['segments']
        for stamp in word_timestamps:
            res.append([stamp['start'], stamp['end'], stamp['segment']])
        df = pd.DataFrame(res, columns=['start', 'end', 'segment'])
        df.to_csv(out_file, index=False)


if __name__ == "__main__":
    extract_files_1()
    extract_files_2()
    extract_vocals_from_audio(0)
    extract_vocals_from_audio(1)
    convert_to_mono(0)
    convert_to_mono(1)
    extract_text_from_vocals(0)
    extract_text_from_vocals(1)
