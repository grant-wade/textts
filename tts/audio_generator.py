import threading
import queue
import numpy as np
import torch
from onnxruntime import InferenceSession
from kokoro.kokoro import phonemize, tokenize

class AudioGenerator:
    def __init__(self, voice_name):
        self.voice_name = voice_name
        self.sentence_queue = queue.Queue(maxsize=20)  # Increased buffer sizes
        self.audio_queue = queue.Queue(maxsize=20)
        self.audio_done_event = threading.Event()
        self.audio_done_event.set()  # Start ready to play
        self.stop_event = threading.Event()
        self.worker_thread = threading.Thread(target=self._worker)
        self.sess = InferenceSession("kokoro/kokoro-v0_19.onnx")
        self.worker_thread.start()

    def _worker(self):
        """Worker thread that processes sentences into audio"""
        while not self.stop_event.is_set():
            try:
                sentence = self.sentence_queue.get(timeout=0.1)
                if sentence is None:  # Sentinel value
                    print("Received sentinel value, stopping worker")  # Debug print
                    break
                    
                # Split and process long sentences
                sentence_parts = self._split_sentence(sentence)
                audio_chunks = []
                
                for part in sentence_parts:
                    phonemes = phonemize(part, self.voice_name[0])
                    tokens = tokenize(phonemes)
                    
                    if tokens:
                        ref_s = torch.load(
                            f"kokoro/voices/{self.voice_name}.pt", weights_only=True
                        )[len(tokens)].numpy()
                        tokens = [[0, *tokens, 0]]
                        audio = self.sess.run(
                            None, dict(tokens=tokens, style=ref_s, speed=np.ones(1, np.float32))
                        )[0]
                        audio_chunks.append(audio)
                
                # Merge audio chunks and add to queue
                if audio_chunks:
                    merged_audio = np.concatenate(audio_chunks)
                    self.audio_queue.put(merged_audio)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error processing sentence: {e}")

    def _split_sentence(self, sentence, max_len=400):
        """Split sentences at reasonable boundaries while preserving meaning"""
        parts = []
        current = sentence.strip()
        
        while len(current) > max_len:
            # Prefer to split at sentence boundaries first
            split_pos = max(
                current.rfind('. ', 0, max_len),
                current.rfind('? ', 0, max_len),
                current.rfind('! ', 0, max_len)
            )
            
            # If no sentence boundary found, look for other whitespace
            if split_pos == -1:
                split_pos = current.rfind(' ', 0, max_len)
            
            # If no whitespace at all, force split
            if split_pos == -1:
                split_pos = max_len
                
            parts.append(current[:split_pos+1].strip())
            current = current[split_pos+1:].lstrip()
            
        if current:
            parts.append(current)
            
        return parts

    def add_sentence(self, sentence):
        """Add a sentence to be processed"""
        self.sentence_queue.put(sentence)

    def get_audio(self):
        """Get generated audio"""
        try:
            return self.audio_queue.get(timeout=0.1)
        except queue.Empty:
            return None

    def stop(self):
        """Stop the worker thread"""
        self.stop_event.set()
        self.sentence_queue.put(None)  # Sentinel value
        self.worker_thread.join()
