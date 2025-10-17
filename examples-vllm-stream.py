import time
import re
import threading
import queue
CHUNK_SIZE = 25
def stream_client(model, history, tools, token2wav=None, output_stream=None, prompt_wav='assets/default_female.wav'):
    
    # import pyaudio
    # import numpy as np
    # audio_player = pyaudio.PyAudio()
    # stream = audio_player.open(format=pyaudio.paInt16,
    #                           channels=1,
    #                           rate=24000,
    #                           output=True,
    #                           frames_per_buffer=512)  # 增加缓冲区大小
    stop_adding_tasks = False
    response = {"tts_content": {"tts_text": '', "tts_audio": ''}, "tool_calls": []}
    buffer = []
    text_buffer = ""
    audio_buffer = []
    is_trans = False
    count = 0
    for line, text, audio in model.stream(history, tools=tools, max_tokens=4096, repetition_penalty=1.05, top_p=0.9, temperature=0.7):
        if len(line.get("tool_calls", [])) > 0:
            if len(response["tool_calls"]) == 0:
                response["tool_calls"] += line["tool_calls"]
            else:
                response["tool_calls"][0]['function']['arguments'] = line["tool_calls"][0]['function']['arguments']
        else:
            if text:
                response["tts_content"]["tts_text"] += text
                print(text, end='', flush=True)

                text_buffer += text
                # 检查是否有句子结束符（句号、问号、感叹号等）
                # if re.search(r'[。！？]$', text):  # 检查当前文本是否以句子结束符结尾
                #     print("\n打印一整句话>>", text_buffer)
                #     text_buffer = ""
                #     is_trans = True

            if audio:
                response["tts_content"]["tts_audio"] += line.get("tts_content", {}).get("tts_audio", '')
                # print("语音token>>", audio, end='', flush=True)
                tts_task_queue.put(audio)
                count += 1
                if count % 100 == 0:
                    print("\n生成100个tokens", count / 100, datetime.now())                
                
                # if output_stream:
                    # buffer += audio
                    # audio_buffer += audio
                    # if len(buffer) >= CHUNK_SIZE + token2wav.flow.pre_lookahead_len:

                    #     start_time = time.time()

                    #     output = token2wav.stream(buffer[:CHUNK_SIZE + token2wav.flow.pre_lookahead_len], prompt_wav=prompt_wav)

                    #     elapsed_time = time.time() - start_time
                    #     # print(f"[执行时间: {elapsed_time:.4f}秒]")

                    #     with open(output_stream, 'ab') as f:
                    #         f.write(output)
                    #     buffer = buffer[CHUNK_SIZE:]
            # if is_trans:
            #     print("\n打印一句话结束")
            #     output = token2wav.stream(audio_buffer, prompt_wav=prompt_wav)
            #     stream.write(output)
            #     audio_buffer = []
            #     is_trans = False

    # if output_stream and len(buffer) > 0:
    #     output = token2wav.stream(buffer, prompt_wav=prompt_wav, last_chunk=True)
    #     with open(output_stream, 'ab') as f:
    #         f.write(output)

    stop_adding_tasks = True
    return response

# === 异步任务队列 ===
tts_task_queue = queue.Queue()  # 存放待合成的文本
playback_queue = queue.Queue()  # 存放已合成的音频数据
stop_adding_tasks = False
if __name__ == "__main__":
    import wave

    from stepaudio2vllm import StepAudio2
    from pathlib import Path
    from token2wav import Token2wav
    import pyaudio
    import sounddevice as sd
    import numpy as np
    from datetime import datetime

    api_url = "http://localhost:8000/v1/chat/completions"
    model_name = "step-audio-2-mini"
    prompt_wav = "assets/default_female.wav"

    model = StepAudio2(api_url, model_name)
    token2wav = Token2wav('/home/promote/.cache/modelscope/hub/models/stepfun-ai/Step-Audio-2-mini/token2wav')
    tokens = [1493, 4299, 4218, 2049, 528, 2752, 4850, 4569, 4575, 6372, 2127, 4068, 2312, 4993, 4769, 2300, 226, 2175, 2160, 2152, 6311, 6065, 4859, 5102, 4615, 6534, 6426, 1763, 2249, 2209, 5938, 1725, 6048, 3816, 6058, 958, 63, 4460, 5914, 2379, 735, 5319, 4593, 2328, 890, 35, 751, 1483, 1484, 1483, 2112, 303, 4753, 2301, 5507, 5588, 5261, 5744, 5501, 2341, 2001, 2252, 2344, 1860, 2031, 414, 4366, 4366, 6059, 5300, 4814, 5092, 5100, 1923, 3054, 4320, 4296, 2148, 4371, 5831, 5084, 5027, 4946, 4946, 2678, 575, 575, 521, 518, 638, 1367, 2804, 3402, 4299]
    token2wav.set_stream_cache(prompt_wav)
    token2wav.stream(tokens[:CHUNK_SIZE + token2wav.flow.pre_lookahead_len], prompt_wav=prompt_wav) # Warm up

    output_stream = Path('output-stream.pcm')
    output_stream.unlink(missing_ok=True)

    history = [
        {"role": "system", "content": "你的名字叫做小跃，是由阶跃星辰公司训练出来的语音大模型。\n你具备调用工具解决问题的能力，你需要根据用户的需求和上下文情景，自主选择是否调用系统提供的工具来协助用户。\n你情感细腻，观察能力强，擅长分析用户的内容，并作出善解人意的回复，说话的过程中时刻注意用户的感受，富有同理心，提供多样的情绪价值。\n今天是2025年8月28日，星期四\n请用默认女声与用户交流"},
        # {"role": "human", "content": [{"type": "audio", "audio": "assets/帮我查一下今天上证指数的开盘价是多少.wav"}]},
        {"role": "human", "content": [{"type": "text", "text": "朗读<<满江红·写怀>>"}]},
        {"role": "assistant", "content": "<tts_start>", "eot": False},
    ]
    tools = [{"type": "function", "function": {"name": "search", "description": "搜索工具", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "搜索关键词"}}, "required": ["query"], "additionalProperties": False}}}]


    # === 播放线程 ===
    audio_player = pyaudio.PyAudio()
    stream = audio_player.open(format=pyaudio.paInt16,
                              channels=1,
                              rate=24000,
                              output=True,
                              frames_per_buffer=1024)  # 增加缓冲区大小 
    def playback_worker():
        while True:
            audio_data = playback_queue.get()
            # if audio_data is None:  # 结束信号
            #     break
            if audio_data:
                start = time.time()
                print("\n语音播放开始时刻", datetime.now())                                
                stream.write(audio_data)
                # print(f"\n[语音播放时间] {time.time() - start:.4f}s") 
                print("\n语音播放完成时刻", datetime.now())               
                # audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
                # sd.play(audio_np, samplerate=24000, blocking=False)
                playback_queue.task_done()
            else:
                time.sleep(0.01)  # 避免忙等待
    # === TTS 合成线程 ===
    def tts_worker():
        while True:
            if stop_adding_tasks and tts_task_queue.empty():
                break

            # 取出最多 50 个 token 进行合成
            batch_tokens = []
            for _ in range(100):
                try:
                    token = tts_task_queue.get(timeout=1)[0]  # 假设 token 结构是 [token_id]
                    batch_tokens.append(token)
                except queue.Empty:
                    break    
            if batch_tokens:
                try:
                    # 合成语音（假设 token2wav.stream 返回 bytes）
                    start = time.time()
                    print("\n语音转换开始时刻和token数", datetime.now(), len(batch_tokens))
                    wav_data = token2wav.stream(batch_tokens, prompt_wav=prompt_wav, last_chunk=True)
                    # print(f"\n[语音合成时间] {time.time() - start:.4f}s tokens = [{len(batch_tokens)}]")
                    print("\n语音转换完成时刻", datetime.now())                    
                    playback_queue.put(wav_data)

                except Exception as e:
                    print(f"[TTS Error] {e}")
                finally:
                    tts_task_queue.task_done()
            else:
                time.sleep(0.01)  # 避免忙等待

            # if tts_task_queue.qsize() >= 100 or (stop_adding_tasks and not tts_task_queue.empty()):
            #     print("\n tts_task_queue长度>>", tts_task_queue.qsize(), stop_adding_tasks)
            #     try:
            #         # 合成语音（假设 token2wav.stream 返回 bytes）
            #         # 将队列转换为数组
            #         tts_task_array = []
            #         while not tts_task_queue.empty():
            #             tts_task_array.append(tts_task_queue.get()[0])
            #         start = time.time()
            #         wav_data = token2wav.stream(tts_task_array, prompt_wav=prompt_wav, last_chunk=True)
            #         print(f"\n[语音合成时间] {time.time() - start:.4f}s")
            #         playback_queue.put(wav_data)

            #     except Exception as e:
            #         print(f"[TTS Error] {e}")
            #     finally:
            #         tts_task_queue.task_done()
            time.sleep(0.01)  # 避免忙等待


    # 启动后台线程
    threading.Thread(target=tts_worker, daemon=True).start()
    threading.Thread(target=playback_worker, daemon=True).start()

    response = stream_client(model, history, tools, token2wav, output_stream, prompt_wav)

    # with open('assets/search_result.txt') as f:
    #     search_result = f.read().strip()
    # history.pop(-1)
    # history += [
    #     {"role": "assistant", "tts_content": response["tts_content"], "tool_calls": response["tool_calls"]},
    #     {"role": "input", "tool_call_id": response["tool_calls"][0]["id"], "content": [{"type": "text", "text": search_result}, {"type": "text", "text": '\n\n\n请用口语化形式总结检索结果，简短地回答用户的问题。'}]},
    #     {"role": "assistant", "content": "<tts_start>", "eot": False},
    # ]
    # response = stream_client(model, history, tools, token2wav, output_stream, prompt_wav)

    # with open(output_stream, 'rb') as f:
    #     pcm = f.read()
    # wav_path = output_stream.with_suffix('.wav')
    # with wave.open(str(wav_path), 'wb') as wf:
    #     wf.setnchannels(1)
    #     wf.setsampwidth(2)
    #     wf.setframerate(24000)
    #     wf.writeframes(pcm)

        # === 结束信号 ===
    tts_task_queue.join()  # 等待所有 TTS 完成
    playback_queue.join()  # 等待所有播放完成

    # 发送结束信号
    tts_task_queue.put(None)
    playback_queue.put(None)

    # 清理资源
    stream.close()
    audio_player.terminate()     
