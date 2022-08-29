import os
import regex as re

from playsound import playsound
from music21 import converter


cwd = os.path.dirname(__file__)

def load_training_data():
    with open(os.path.join(cwd, 'data', 'irish.abc'), 'r') as f:
        text = f.read()
    songs = extract_song_snippet(text)
    return songs

def extract_song_snippet(text):
    pattern = '(^|\n\n)(.*?)\n\n'
    search_results = re.findall(pattern, text, overlapped=True, flags=re.DOTALL)
    songs = [song[1] for song in search_results]
    return songs

def save_song_to_abc(song, filename='tmp'):
    save_name = f'{filename}.abc'
    with open(save_name, 'w') as f:
        f.write(song)
    return filename

def abc2wav(abc_file):
    path_to_tool = os.path.join(cwd, 'bin', 'abc2wav')
    return os.system(f'{path_to_tool} {abc_file}')

def play_wav(wav_file):
    return os.system(f'audacious {wav_file}')

def play_song(song, save='tmp'):
    basename = save_song_to_abc(song, filename=save)
    ret = abc2wav(f'{basename}.abc')
    if ret == 0:
        return play_wav(f'{basename}.wav')
    return None

def play_generated_song(generated_text):
    songs = extract_song_snippet(generated_text)
    if len(songs) == 0:
        print("No valid songs found in generated text. Try training the \
        model longer or increasing the amount of generated music to \
        ensure complete songs are generated!")
        return

    for i, song in enumerate(songs):
        play_song(song, save=f'bach_{i}')
