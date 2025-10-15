import re
import time

CHUNK_SIZE = 25

def sentence_level_stream_client(model, history, tools, token2wav=None, output_stream=None, prompt_wav='assets/default_female.wav', play_audio=False):
    """
    句子级别的流式客户端，每生成完一个句子就播放对应的音频，避免卡顿
    """
    # 如果需要播放音频，初始化音频流
    if play_audio:
        import pyaudio
        import numpy as np
        audio_player = pyaudio.PyAudio()
        # 增加缓冲区大小以减少underrun错误
        stream = audio_player.open(format=pyaudio.paInt16,
                                  channels=1,
                                  rate=24000,
                                  output=True,
                                  frames_per_buffer=1024)  # 从512增加到1024
        # 创建一个更大的silence缓冲区用于紧急填充
        silence_chunk = b'\x00\x00' * 2048  # 从1024增加到2048个16-bit的静音样本
    
    response = {"tts_content": {"tts_text": '', "tts_audio": ''}, "tool_calls": []}
    buffer = []
    
    # 句子级别的缓存
    sentence_buffer = []
    sentence_text = ""
    
    print("开始流式生成...")
    
    for line, text, audio in model.stream(history, tools=tools, max_tokens=4096, repetition_penalty=1.05, top_p=0.9, temperature=0.7):
        if len(line.get("tool_calls", [])) > 0:
            if len(response["tool_calls"]) == 0:
                response["tool_calls"] += line["tool_calls"]
            else:
                response["tool_calls"][0]['function']['arguments'] = line["tool_calls"][0]['function']['arguments']
        else:
            if text:
                response["tts_content"]["tts_text"] += text
                sentence_text += text
                print(text, end='', flush=True)
                
                # 检查是否有句子结束符（句号、问号、感叹号等）
                if re.search(r'[。！？.!?]$', text):  # 检查当前文本是否以句子结束符结尾
                    print("\n[检测到句子结束，处理音频播放...]")
                    
                    # 如果有句子结束，处理当前句子的音频
                    if sentence_buffer:
                        # 处理句子缓存中的音频数据
                        while len(sentence_buffer) >= CHUNK_SIZE + token2wav.flow.pre_lookahead_len:
                            start_time = time.time()
                            # 添加speed参数以控制语速
                            output = token2wav.stream(sentence_buffer[:CHUNK_SIZE + token2wav.flow.pre_lookahead_len], prompt_wav=prompt_wav, speed=1.0)
                            elapsed_time = time.time() - start_time
                            print(f"[执行时间: {elapsed_time:.4f}秒]")
                            with open(output_stream, 'ab') as f:
                                f.write(output)
                            # 如果需要播放音频，实时播放
                            if play_audio:
                                try:
                                    stream.write(output)
                                except Exception as e:
                                    print(f"音频播放错误: {e}")
                                    # 发送更多静音数据以保持流活动
                                    stream.write(silence_chunk)
                                    stream.write(silence_chunk)
                        buffer = buffer[CHUNK_SIZE:]
    
    # 处理最后一句（可能没有结束符）
    if sentence_buffer:
        print("\n[处理最后一句...]")
        # 处理句子缓存中的音频数据
        while len(sentence_buffer) >= CHUNK_SIZE + token2wav.flow.pre_lookahead_len:
            start_time = time.time()
            # 添加speed参数以控制语速
            output = token2wav.stream(sentence_buffer[:CHUNK_SIZE + token2wav.flow.pre_lookahead_len], prompt_wav=prompt_wav, speed=1.0)
            elapsed_time = time.time() - start_time
            print(f"[执行时间: {elapsed_time:.4f}秒]")
            with open(output_stream, 'ab') as f:
                f.write(output)
            # 如果需要播放音频，实时播放
            if play_audio:
                try:
                    stream.write(output)
                except Exception as e:
                    print(f"音频播放错误: {e}")
                    # 发送静音数据以保持流活动
                    stream.write(silence_chunk)
            sentence_buffer = sentence_buffer[CHUNK_SIZE:]
            
        # 将剩余的音频数据也处理掉
        if sentence_buffer:
            start_time = time.time()
            # 添加speed参数以控制语速
            output = token2wav.stream(sentence_buffer, prompt_wav=prompt_wav, last_chunk=True, speed=1.0)
            elapsed_time = time.time() - start_time
            print(f"[执行时间: {elapsed_time:.4f}秒]")
            with open(output_stream, 'ab') as f:
                f.write(output)
            # 如果需要播放音频，播放最后一段
            if play_audio:
                try:
                    stream.write(output)
                except Exception as e:
                    print(f"音频播放错误: {e}")
        print("[最后一句处理完成]")

    # 处理剩余的音频数据
    if output_stream and len(buffer) > 0:
        start_time = time.time()
        # 添加speed参数以控制语速
        output = token2wav.stream(buffer, prompt_wav=prompt_wav, last_chunk=True, speed=1.0)
        elapsed_time = time.time() - start_time
        print(f"[执行时间: {elapsed_time:.4f}秒]")
        with open(output_stream, 'ab') as f:
            f.write(output)
        # 如果需要播放音频，播放最后一段
        if play_audio:
            try:
                stream.write(output)
            except Exception as e:
                print(f"音频播放错误: {e}")

    # 关闭音频流
    if play_audio:
        # 在关闭前播放一小段静音以确保缓冲区清空
        try:
            stream.write(silence_chunk)
            time.sleep(0.1)  # 给一点时间让缓冲区播放完毕
        except:
            pass
        stream.stop_stream()
        stream.close()
        audio_player.terminate()
        
    return response

if __name__ == "__main__":
    import wave
    from pathlib import Path

    from stepaudio2vllm import StepAudio2
    from token2wav import Token2wav

    api_url = "http://localhost:8999/v1/chat/completions"
    model_name = "step-audio-2-mini"
    prompt_wav = "assets/default_female.wav"

    model = StepAudio2(api_url, model_name)
    token2wav = Token2wav('/home/promote/.cache/modelscope/hub/models/stepfun-ai/Step-Audio-2-mini/token2wav')
    tokens = [1493, 4299, 4218, 2049, 528, 2752, 4850, 4569, 4575, 6372, 2127, 4068, 2312, 4993, 4769, 2300, 226, 2175, 2160, 2152, 6311, 6065, 4859, 5102, 4615, 6534, 6426, 1763, 2249, 2209, 5938, 1725, 6048, 3816, 6058, 958, 63, 4460, 5914, 2379, 735, 5319, 4593, 2328, 890, 35, 751, 1483, 1484, 1483, 2112, 303, 4753, 2301, 5507, 5588, 5261, 5744, 5501, 2341, 2001, 2252, 2344, 1860, 2031, 414, 4366, 4366, 6059, 5300, 4814, 5092, 5100, 1923, 3054, 4320, 4296, 2148, 4371, 5831, 5084, 5027, 4946, 4946, 2678, 575, 575, 521, 518, 638, 1367, 2804, 3402, 4299]
    token2wav.set_stream_cache(prompt_wav)
    token2wav.stream(tokens[:CHUNK_SIZE + token2wav.flow.pre_lookahead_len], prompt_wav=prompt_wav) # Warm up

    output_stream = Path('output-sentence-stream.pcm')
    output_stream.unlink(missing_ok=True)

    history = [
        {"role": "system", "content": "你的名字叫做小跃，是由阶跃星辰公司训练出来的语音大模型。\n你具备调用工具解决问题的能力，你需要根据用户的需求和上下文情景，自主选择是否调用系统提供的工具来协助用户。\n你情感细腻，观察能力强，擅长分析用户的内容，并作出善解人意的回复，说话的过程中时刻注意用户的感受，富有同理心，提供多样的情绪价值。\n今天是2025年8月28日，星期四\n请用默认女声与用户交流"},
        {"role": "human", "content": [{"type": "audio", "audio": "assets/帮我查一下今天上证指数的开盘价是多少.wav"}]},
        {"role": "assistant", "content": "<tts_start>", "eot": False},
    ]
    tools = [{"type": "function", "function": {"name": "search", "description": "搜索工具", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "搜索关键词"}}, "required": ["query"], "additionalProperties": False}}}]

    # 使用句子级别的流式播放
    response = sentence_level_stream_client(model, history, tools, token2wav, output_stream, prompt_wav, play_audio=True)
    
    print("\n\n完整响应:")
    print(response)