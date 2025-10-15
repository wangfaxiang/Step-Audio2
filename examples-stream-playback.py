#!/usr/bin/env python3
"""
流式语音播放示例
该示例演示了如何将生成的语音实时播放到耳机中
"""

import pyaudio
import numpy as np
import wave
from pathlib import Path
import time

from stepaudio2vllm import StepAudio2
from token2wav import Token2wav

CHUNK_SIZE = 25

def stream_client_with_playback(model, history, tools, token2wav=None, output_stream=None, 
                               prompt_wav='assets/default_female.wav'):
    """
    流式客户端，支持实时播放到耳机
    
    Args:
        model: 模型对象
        history: 对话历史
        tools: 工具列表
        token2wav: Token到WAV转换器
        output_stream: 输出流文件路径
        prompt_wav: 提示语音文件路径
    """
    
    # 初始化音频播放设备
    audio_player = pyaudio.PyAudio()
    
    # 增加缓冲以减少underrun错误
    stream = audio_player.open(format=pyaudio.paInt16,
                              channels=1,
                              rate=24000,  # 与token2wav中采样率一致
                              output=True,
                              frames_per_buffer=512)  # 增加缓冲区大小
    
    response = {"tts_content": {"tts_text": '', "tts_audio": ''}, "tool_calls": []}
    buffer = []
    
    # 创建一个silence缓冲区用于紧急填充
    silence_chunk = b'\x00\x00' * 1024  # 1024个16-bit的静音样本
    
    try:
        for line, text, audio in model.stream(history, tools=tools, max_tokens=4096, 
                                             repetition_penalty=1.05, top_p=0.9, temperature=0.7):
            if len(line.get("tool_calls", [])) > 0:
                if len(response["tool_calls"]) == 0:
                    response["tool_calls"] += line["tool_calls"]
                else:
                    response["tool_calls"][0]['function']['arguments'] = line["tool_calls"][0]['function']['arguments']
            else:
                if text:
                    response["tts_content"]["tts_text"] += text
                    print(text, end='', flush=True)
                if audio:
                    response["tts_content"]["tts_audio"] += line.get("tts_content", {}).get("tts_audio", '')
                    print(audio, end='', flush=True)
                    
                    # 处理音频流
                    buffer += audio
                    if len(buffer) >= CHUNK_SIZE + token2wav.flow.pre_lookahead_len:
                        # 转换音频数据
                        start_time = time.time()
                        output = token2wav.stream(buffer[:CHUNK_SIZE + token2wav.flow.pre_lookahead_len], 
                                                 prompt_wav=prompt_wav)
                        elapsed_time = time.time() - start_time
                        print(f"[执行时间: {elapsed_time:.4f}秒]")
                        
                        # 保存到文件（如果指定了输出路径）
                        if output_stream:
                            with open(output_stream, 'ab') as f:
                                f.write(output)
                        
                        # 实时播放到耳机 - 添加错误处理
                        try:
                            stream.write(output)
                        except Exception as e:
                            print(f"音频播放错误: {e}")
                            # 发送静音数据以保持流活动
                            stream.write(silence_chunk)
                        
                        # 更新缓冲区
                        buffer = buffer[CHUNK_SIZE:]
                        
        # 处理剩余的音频数据
        if len(buffer) > 0:
            start_time = time.time()
            output = token2wav.stream(buffer, prompt_wav=prompt_wav, last_chunk=True)
            elapsed_time = time.time() - start_time
            print(f"[执行时间: {elapsed_time:.4f}秒]")
            
            if output_stream:
                with open(output_stream, 'ab') as f:
                    f.write(output)
            
            # 播放剩余音频
            try:
                stream.write(output)
            except Exception as e:
                print(f"音频播放错误: {e}")
            
    finally:
        # 在关闭前播放一小段静音以确保缓冲区清空
        try:
            stream.write(silence_chunk)
            time.sleep(0.1)  # 给一点时间让缓冲区播放完毕
        except:
            pass
            
        # 清理音频资源
        stream.stop_stream()
        stream.close()
        audio_player.terminate()
        
    return response

if __name__ == "__main__":
    api_url = "http://localhost:8999/v1/chat/completions"
    model_name = "step-audio-2-mini"
    prompt_wav = "assets/default_female.wav"

    model = StepAudio2(api_url, model_name)
    token2wav = Token2wav('/home/promote/.cache/modelscope/hub/models/stepfun-ai/Step-Audio-2-mini/token2wav')
    
    # 预热模型
    tokens = [1493, 4299, 4218, 2049, 528, 2752, 4850, 4569, 4575, 6372, 2127, 4068, 2312, 4993, 4769, 2300, 226, 2175, 2160, 2152, 6311, 6065, 4859, 5102, 4615, 6534, 6426, 1763, 2249, 2209, 5938, 1725, 6048, 3816, 6058, 958, 63, 4460, 5914, 2379, 735, 5319, 4593, 2328, 890, 35, 751, 1483, 1484, 1483, 2112, 303, 4753, 2301, 5507, 5588, 5261, 5744, 5501, 2341, 2001, 2252, 2344, 1860, 2031, 414, 4366, 4366, 6059, 5300, 4814, 5092, 5100, 1923, 3054, 4320, 4296, 2148, 4371, 5831, 5084, 5027, 4946, 4946, 2678, 575, 575, 521, 518, 638, 1367, 2804, 3402, 4299]
    token2wav.set_stream_cache(prompt_wav)
    token2wav.stream(tokens[:CHUNK_SIZE + token2wav.flow.pre_lookahead_len], prompt_wav=prompt_wav)  # Warm up

    # 设置输出文件
    output_stream = Path('output-stream.pcm')
    output_stream.unlink(missing_ok=True)

    # 构建对话历史
    history = [
        {"role": "system", "content": "你的名字叫做小跃，是由阶跃星辰公司训练出来的语音大模型。\n你具备调用工具解决问题的能力，你需要根据用户的需求和上下文情景，自主选择是否调用系统提供的工具来协助用户。\n你情感细腻，观察能力强，擅长分析用户的内容，并作出善解人意的回复，说话的过程中时刻注意用户的感受，富有同理心，提供多样的情绪价值。\n今天是2025年8月28日，星期四\n请用默认女声与用户交流"},
        {"role": "human", "content": [{"type": "audio", "audio": "assets/帮我查一下今天上证指数的开盘价是多少.wav"}]},
        {"role": "assistant", "content": "<tts_start>", "eot": False},
    ]
    
    # 定义工具
    tools = [{"type": "function", "function": {"name": "search", "description": "搜索工具", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "搜索关键词"}}, "required": ["query"], "additionalProperties": False}}}]

    # 第一轮对话
    print("开始流式语音播放...")
    response = stream_client_with_playback(model, history, tools, token2wav, output_stream, prompt_wav)

    # 处理工具调用结果
    with open('assets/search_result.txt') as f:
        search_result = f.read().strip()
        
    history.pop(-1)
    history += [
        {"role": "assistant", "tts_content": response["tts_content"], "tool_calls": response["tool_calls"]},
        {"role": "input", "tool_call_id": response["tool_calls"][0]["id"], "content": [{"type": "text", "text": search_result}, {"type": "text", "text": '\n\n\n请用口语化形式总结检索结果，简短地回答用户的问题。'}]},
        {"role": "assistant", "content": "<tts_start>", "eot": False},
    ]
    
    # 第二轮对话
    response = stream_client_with_playback(model, history, tools, token2wav, output_stream, prompt_wav)

    # 保存为WAV文件
    with open(output_stream, 'rb') as f:
        pcm = f.read()
    wav_path = output_stream.with_suffix('.wav')
    with wave.open(str(wav_path), 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(24000)
        wf.writeframes(pcm)
        
    print("\n\n流式语音播放完成！")