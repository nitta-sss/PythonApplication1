import sys
sys.path.append("C:/Users/232144/Desktop/HaLu/src/Audio/")
sys.path.append("C:/Users/232144/Desktop/HaLu/data/")

import Voice_Read
import Emotional_Reasoning
text=Voice_Read.start_voice_read()

val, aro=Emotional_Reasoning.suiron_test(text)

print("音声認識：",text)
print("感情グラフ\n快ー不快\n覚醒ー静寂",val,aro)

