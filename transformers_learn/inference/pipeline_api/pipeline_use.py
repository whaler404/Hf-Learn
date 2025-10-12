from transformers import pipeline

model="facebook/wav2vec2-base-960h"
transcriber = pipeline(task="automatic-speech-recognition", 
                       model=model,
                       device=1,
                       return_timestamps="word",
                       )

result = transcriber("./datasets/mlk.flac")
print(result)