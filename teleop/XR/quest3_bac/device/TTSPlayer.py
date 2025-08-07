import asyncio
from pathlib import Path
from typing import Text

from kokoro_onnx import Kokoro
from misaki import zh
from openai import OpenAI
import pyaudio
import sounddevice as sd

from teleop.XR.quest3.utils.log import logger


class TTSPlayerFastAPI:
    def __init__(self, url: Text = "http://localhost:8880/v1", language: Text = "zh-CN"):
        self.client = OpenAI(base_url=url, api_key="not-needed")
        self.player = pyaudio.PyAudio().open(format=pyaudio.paInt16, channels=1, rate=24000, output=True)
        self.language = language
        self._speaking_task = None

    def play(self, message: Text):
        """Play the given message using TTS.

        Args:
            message (Text): The message to be played.
        """
        if self.language == "zh-CN":
            with self.client.audio.speech.with_streaming_response.create(
                    model="kokoro",
                    # voice="zf_xiaoxiao(1)+zf_xiaobei(1)",
                    voice="zf_xiaoyi",
                    response_format="pcm",
                    input=f"{message}",
            ) as response:
                for chunk in response.iter_bytes(chunk_size=1024):
                    self.player.write(chunk)
        else:
            raise NotImplemented(f"Language {self.language} not supported yet.")

    async def speak(self, message: Text):
        """Asynchronously play the given message using TTS.

        Args:
            message (Text): The message to be played.
        """
        # Cancel any ongoing speech
        if self._speaking_task:
            self._speaking_task.cancel()
            await asyncio.shield(self._speaking_task)
        # Start new speaking task
        self._speaking_task = asyncio.create_task(self._play_async(message))
        return self._speaking_task

    async def _play_async(self, message: Text):
        """Internal async method to play TTS without blocking.

        Args:
            message (Text): The message to be played.
        """
        if self.language == "zh-CN":
            try:
                with self.client.audio.speech.with_streaming_response.create(
                        model="kokoro",
                        voice="zf_xiaoyi",
                        response_format="pcm",
                        input=f"{message}",
                ) as response:
                    for chunk in response.iter_bytes(chunk_size=1024):
                        # Run blocking audio write in a separate thread
                        await asyncio.to_thread(self.player.write, chunk)
            except asyncio.CancelledError:
                # Handle cancellation gracefully
                pass
        else:
            raise NotImplementedError(f"Language {self.language} not supported yet.")


class TTSPlayer:
    def __init__(self):
        model_path = Path(__file__).parent.parent.joinpath("models")
        self.player = pyaudio.PyAudio().open(format=pyaudio.paInt16, channels=1, rate=24000, output=True)
        self._speaking_task = None
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model path {model_path} does not exist.Please download the models under the path.")
        self.language = "zh-CN"
        if self.language == "zh-CN":
            self.client = Kokoro(model_path=f"{model_path.joinpath('kokoro-v1.1-zh.onnx')}",
                                 voices_path=f"{model_path.joinpath('voices-v1.1-zh.bin')}",
                                 vocab_config=f"{model_path.joinpath('config.json')}")
            if self.language == "zh-CN":
                self.g2p = zh.ZHG2P(version="1.1")
        else:
            raise NotImplementedError(f"Language {self.language} not supported yet.")

    async def speak(self, message: Text):
        """Asynchronously play the given message using TTS.

        Args:
            message (Text): The message to be played.
        """
        if self._speaking_task:
            sd.stop()
            self._speaking_task.cancel()
            await asyncio.shield(self._speaking_task)
            logger.warning("Cancelled ongoing speech task.")
        self._speaking_task = asyncio.create_task(self._play_async(message))
        return self._speaking_task

    async def _play_async(self, message: Text):
        """ Asynchronously play TTS without blocking.

        Args:
            message (Text): The message to be played.
        """
        if self.language == "zh-CN":
            try:
                phonemes, _ = self.g2p(message)
                # ['af_maple', 'af_sol', 'bf_vale', 'zf_001', 'zf_002', 'zf_003', 'zf_004', 'zf_005', 'zf_006', 'zf_007',
                #  'zf_008', 'zf_017', 'zf_018', 'zf_019', 'zf_021', 'zf_022', 'zf_023', 'zf_024', 'zf_026', 'zf_027',
                #  'zf_028', 'zf_032', 'zf_036', 'zf_038', 'zf_039', 'zf_040', 'zf_042', 'zf_043', 'zf_044', 'zf_046',
                #  'zf_047', 'zf_048', 'zf_049', 'zf_051', 'zf_059', 'zf_060', 'zf_067', 'zf_070', 'zf_071', 'zf_072',
                #  'zf_073', 'zf_074', 'zf_075', 'zf_076', 'zf_077', 'zf_078', 'zf_079', 'zf_083', 'zf_084', 'zf_085',
                #  'zf_086', 'zf_087', 'zf_088', 'zf_090', 'zf_092', 'zf_093', 'zf_094', 'zf_099']
                samples, sample_rate = self.client.create(phonemes, voice="zf_077", speed=1.8, is_phonemes=True)
                sd.play(samples, samplerate=sample_rate)
                # sd.wait()
            except asyncio.CancelledError:
                sd.stop()
                pass
        else:
            raise NotImplementedError(f"Language {self.language} not supported yet.")


if __name__ == "__main__":
    # 末端移动速度设置为 0.01 米每秒
    tts = TTSPlayer()
    asyncio.run(tts.speak("你好，我是你的助手 蒙 特 0 1"))
    # tts.speak("你好，我是你的助手 蒙 特 0 1")
