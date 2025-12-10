import sys
sys.path.append("C:/Users/232144/Desktop/HaLu/src/Audio/")
sys.path.append("C:/Users/232144/Desktop/HaLu/data/")

import Voice_Read
import Emotional_Reasoning

text=Voice_Read.start_voice_read()

Emotional_Reasoning.suiron_test(text)


