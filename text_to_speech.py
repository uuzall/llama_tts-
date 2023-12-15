import torch 

def tts(processor, model, vocoder, text, speaker_embeddings): 
  inputs = processor(text=text, return_tensors="pt")
  with torch.no_grad(): 
    speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)
  return speech

