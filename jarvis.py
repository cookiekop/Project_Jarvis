from audiobox import AudioBox
from cvbox import CVbox
from clip import Clip
from moviepy.editor import *
import numpy as np
from configurations import *
import cv2

class Jarvis:
    def __init__(self):
        self._vision = CVbox()
        self._hearing = AudioBox()
        self.current_clip = None

jarvis = Jarvis()
videoclip = VideoFileClip('samples/sample.mp4', target_resolution=RESOLUTION, audio_fps=AUDIO_FPS)
audio = videoclip.audio.to_soundarray() * 32768.0
if len(audio.shape) > 1:
    audio = np.mean(audio, axis=1)
audio = audio.astype(np.int16)
new_clip = Clip(audio)
video = videoclip.without_audio()
for i, frame in enumerate(videoclip.iter_frames()):
    if i % CV_BOX_INTERVAL == 0:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        sce = jarvis._vision.run(frame)
        new_clip.scenes.append(sce)

sounds, words = jarvis._hearing.run(audio)
new_clip.sounds = sounds
new_clip.narative = words
new_clip.save()
