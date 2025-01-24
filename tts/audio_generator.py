import threading
import queue
import time
import numpy as np
import torch
from onnxruntime import InferenceSession
from kokoro.kokoro import phonemize, tokenize

class BaseAudioGenerator:
    def __init__(self, voice_name, speed=1.0):
        self.voice_name = voice_name
        self.speed = speed
        self.sess = InferenceSession("kokoro/kokoro-v0_19.onnx")


    def _generate_audio_for_part(self, part):
        """Generate audio for a single text part"""
        phonemes = phonemize(part, self.voice_name[0])
        tokens = tokenize(phonemes)

        if tokens:
            ref_s = torch.load(
                f"kokoro/voices/{self.voice_name}.pt", weights_only=True
            )[len(tokens)].numpy()
            tokens = [[0, *tokens, 0]]
            return self.sess.run(
                None,
                dict(
                    tokens=tokens, 
                    style=ref_s, 
                    speed=np.ones(1, np.float32) * self.speed
                ),
            )[0]
        return None



class AudioGenerator(BaseAudioGenerator):
    def __init__(self, voice_name):
        super().__init__(voice_name)
        self.sentence_queue = queue.Queue(maxsize=20)  # Increased buffer sizes
        self.audio_queue = queue.Queue(maxsize=20)
        self.audio_done_event = threading.Event()
        self.audio_done_event.set()  # Start ready to play
        self.stop_event = threading.Event()
        self.processing_complete = False
        self.worker_thread = threading.Thread(target=self._worker)
        self.worker_thread.start()

    def _worker(self):
        """Worker thread that processes sentences into audio"""
        while not self.stop_event.is_set():
            try:
                sentence = self.sentence_queue.get(timeout=0.1)
                if self.stop_event.is_set():  # Immediate check after getting item
                    break

                # Split and process long sentences
                sentence_parts = self._split_sentence(sentence)
                audio_chunks = []

                for part in sentence_parts:
                    if self.stop_event.is_set():  # Check before each part
                        break
                    audio = self._generate_audio_for_part(part)
                    if audio is not None:
                        audio_chunks.append(audio)

                # Merge audio chunks and add to queue
                if audio_chunks:
                    merged_audio = np.concatenate(audio_chunks)
                    self.audio_queue.put(merged_audio)
                    
                # Mark as complete if queue is empty
                if self.sentence_queue.empty():
                    self.processing_complete = True

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error processing sentence: {e}")
                self.processing_complete = True

    def _split_sentence(self, sentence, max_len=400):
        """Split sentences at reasonable boundaries while preserving meaning"""
        parts = []
        current = sentence.strip()

        while len(current) > max_len:
            # Prefer to split at sentence boundaries first
            split_pos = max(
                current.rfind(". ", 0, max_len),
                current.rfind("? ", 0, max_len),
                current.rfind("! ", 0, max_len),
            )

            # If no sentence boundary found, look for other whitespace
            if split_pos == -1:
                split_pos = current.rfind(" ", 0, max_len)

            # If no whitespace at all, force split
            if split_pos == -1:
                split_pos = max_len

            parts.append(current[: split_pos + 1].strip())
            current = current[split_pos + 1 :].lstrip()

        if current:
            parts.append(current)

        return parts

    def add_sentence(self, sentence):
        """Add a sentence to be processed"""
        # Non-blocking put with size check
        if self.sentence_queue.qsize() < self.sentence_queue.maxsize:
            self.sentence_queue.put(sentence)
        else:
            time.sleep(0.1)  # Wait before retrying

    def get_audio(self):
        """Get generated audio"""
        try:
            return self.audio_queue.get(timeout=0.1)
        except queue.Empty:
            return None

    def is_processing(self):
        """Check if there's pending work or audio"""
        return not self.sentence_queue.empty() or not self.audio_queue.empty()

    def has_pending_sentences(self):
        """Check if there are sentences waiting to be processed"""
        return not self.sentence_queue.empty()

    def stop(self):
        """Stop the worker thread and ensure all audio is processed"""
        # Wait for processing to complete
        start_wait = time.time()
        while not self.processing_complete and time.time() - start_wait < 5.0:
            time.sleep(0.1)
            
        self.stop_event.set()
        
        # Clear queues to break potential blocking
        while not self.sentence_queue.empty():
            try:
                self.sentence_queue.get_nowait()
            except queue.Empty:
                break
                
        if self.worker_thread.is_alive():
            # Wait longer for final audio processing
            self.worker_thread.join(timeout=5.0)
            if self.worker_thread.is_alive():
                print("Warning: Audio worker thread did not terminate cleanly")
            else:
                # Ensure any remaining audio is processed
                while not self.audio_queue.empty():
                    try:
                        self.audio_queue.get_nowait()
                    except queue.Empty:
                        break

class AudioGeneratorSync(BaseAudioGenerator):
    def __init__(self, voice_name, sample_rate=22050, volume=1.0, speed=1.0):
        super().__init__(voice_name, speed)
        self.sample_rate = sample_rate
        self.volume = volume
        self.current_audio = None

    def add_sentence(self, sentence):
        """Add a sentence to generate audio for"""
        sentence_parts = self._split_sentence(sentence)
        audio_chunks = []

        for part in sentence_parts:
            audio = self._generate_audio_for_part(part)
            if audio is not None:
                audio_chunks.append(audio)

        if audio_chunks:
            self.current_audio = np.concatenate(audio_chunks)
        else:
            self.current_audio = None

    def get_audio(self):
        """Get the generated audio data"""
        return self.current_audio
