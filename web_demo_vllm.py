import tempfile
import traceback
from pathlib import Path

import gradio as gr
import json
import time
from datetime import datetime
import threading
import queue
import pyaudio
def save_tmp_audio(audio_bytes, cache_dir):
    with tempfile.NamedTemporaryFile(dir=cache_dir, delete=False, suffix=".wav") as temp_audio:
        temp_audio.write(audio_bytes)
    return temp_audio.name


def add_message(chatbot, history, mic, text):
    if not mic and not text:
        return chatbot, history, "Input is empty"

    if text:
        chatbot.append({"role": "user", "content": text})
        history.append({"role": "human", "content": text})
    elif mic and Path(mic).exists():
        chatbot.append({"role": "user", "content": {"path": mic}})
        history.append({"role": "human", "content": [{"type": "audio", "audio": mic}]})

    return chatbot, history, None


def reset_state(system_prompt):
    return [], [{"role": "system", "content": system_prompt}]

# 定义开启空调的工具函数
def turn_on_air_conditioner(temperature: int = 26, mode: str = "auto"):
    """
    开启智能座舱内空调并设置指定温度和模式
    
    参数:
        temperature (int): 设定温度，默认26摄氏度
        mode (str): 运行模式，可选：'cool'（制冷）、'heat'（制热）、'fan'（送风）、'auto'（自动）
    
    返回:
        str: 操作结果提示
    """
    # 这里可以接入真实的智能家居控制API
    # 例如通过Wi-Fi或蓝牙控制空调设备
    # 当前为模拟实现
    supported_modes = ['cool', 'heat', 'fan', 'auto']
    if mode not in supported_modes:
        return f"抱歉，不支持的模式：{mode}。可选模式有：{', '.join(supported_modes)}"
    
    # 模拟发送指令到空调设备
    print(f"[调试信息] 正在向空调发送开机指令... 温度: {temperature}°C, 模式: {mode}")
    
    return f"空调设定完成，设定温度为{temperature}度，运行模式为{mode}。"

# 定义开启空调的工具函数
def turn_on_play_music(mode: str = "list_loop"):
    """
    开启音乐播放器并设置播放模式
    
    参数:
        mode (str): 播放模式，可选：'single_repeat'（单曲循环播放）、'list_loop'（列表循环播放）、'random'（随机播放）
    
    返回:
        str: 操作结果提示
    """
    # 这里可以接入真实的智能家居控制API
    # 例如通过Wi-Fi或蓝牙控制空调设备
    # 当前为模拟实现
    supported_modes = ['list_loop', 'single_repeat', 'random']
    if mode not in supported_modes:
        return f"抱歉，不支持的操作模式：{mode}。可选模式有：{', '.join(supported_modes)}"
    
    # 模拟发送指令到空调设备
    print(f"[调试信息] 正在打开音乐播放器...")
    
    if mode == 'single_repeat':
        return "已经为您打开音乐播放器，模式为单曲循环播放。"
    elif mode == 'list_loop':
        return "已经为您打开音乐播放器，模式为列表循环播放。"
    elif mode == 'random':
        return "已经为您打开音乐播放器，模式为随机播放。"
    else:
        return "抱歉，不支持的操作模式：{}".format(mode)

# === 工具映射表：把名字映射到实际函数 ===
AVAILABLE_TOOLS = {
    "turn_on_air_conditioner": turn_on_air_conditioner,
    "turn_on_play_music": turn_on_play_music,
}

def predict(chatbot, history, audio_model, token2wav, prompt_wav, cache_dir):
    try:
        # Request speech response
        history.append({"role": "assistant", "content": "<tts_start>", "eot": False})

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "turn_on_air_conditioner",
                    "description": "开启智能座舱内空调并设置指定温度和模式",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "temperature": {"type": "integer", "description": "目标温度，16-30℃"},
                            "mode": {"type": "string", "enum": ["cool", "heat", "fan", "auto"], "description": "模式"}
                        },
                        "required": ["temperature", "mode"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "turn_on_play_music",
                    "description": "开启音乐播放器并设置播放模式",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "mode": {"type": "string", "enum": ["single_repeat", "list_loop","random"], "description": "播放模式"}
                        },
                        "required": ["mode"]
                    }
                }
            }
        ]

        response, text, audio = audio_model(
            history,
            tools=tools,
            max_tokens=2048,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.05,
        )
        # Convert audio tokens to waveform and append to chat
        if audio:
            audio = [x for x in audio if x < 6561]
            audio_bytes = token2wav(audio, prompt_wav)
            audio_path = save_tmp_audio(audio_bytes, cache_dir)
            chatbot.append({"role": "assistant", "content": {"path": audio_path}})
            # history[-1] = {"role": "assistant", "tts_content": response["tts_content"]}
            history[-1] = {"role": "assistant", "content": text}            
        else:
            chatbot.append({"role": "assistant", "content": text})
            history[-1] = {"role": "assistant", "content": text}

        # === 处理工具调用 ===
        if response.get("tool_calls"):
            tool_calls = response["tool_calls"]
            tool_call = tool_calls[0]  # 假设只调一个
            tool_name = tool_call["function"]["name"]
            try:
                args = json.loads(tool_call["function"]["arguments"])  # 解析参数
            except json.JSONDecodeError as e:
                print("参数解析失败:", e)
                return

            # === ★ 核心：判断并调用本地函数 ★ ===
            if tool_name in AVAILABLE_TOOLS:
                # 执行本地函数
                tool_result = AVAILABLE_TOOLS[tool_name](**args)

                # 更新 history：先删掉原来的 assistant 请求
                # history.pop(-1)

                # 添加完整交互链
                history += [
                    # 模型发起工具调用
                    {
                        "role": "assistant",
                        "tool_calls": tool_calls
                    },
                    # 工具执行结果（模拟从设备/API 返回）
                    {
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": tool_result  # ← 这是 turn_on_air_conditioner 的返回值
                    },
                    # 模型继续生成自然语言回复
                    {
                        "role": "assistant",
                        "content": "<tts_start>",
                        "eot": False
                    }
                ]

                # 第二次调用：模型基于工具结果生成语音回复
                print("历史会话>>", history)
                final_response, final_text, final_audio = audio_model(
                    history, tools=tools, max_tokens=512,
                    repetition_penalty=1.05, top_p=0.9, temperature=0.7
                )

                if final_audio:
                    final_audio = [x for x in final_audio if x < 6561]
                    audio_bytes = token2wav(final_audio, prompt_wav)
                    audio_path = save_tmp_audio(audio_bytes, cache_dir)
                    chatbot.append({"role": "assistant", "content": {"path": audio_path}})
                    # history[-1] = {"role": "assistant", "tts_content": response["tts_content"]}
                    history[-1] = {"role": "assistant", "content": text}
                else:
                    chatbot.append({"role": "assistant", "content": text})
                    history[-1] = {"role": "assistant", "content": text}

                print("最终回复:", final_text)
                # if final_audio:
                #     final_audio = [x for x in final_audio if x < 6561]
                #     wav_data = token2wav(final_audio, prompt_wav='assets/default_female.wav')
                #     with open('output-tool-call-success.wav', 'wb') as f:
                #         f.write(wav_data)
                #     print("✅ 音频已生成：output-tool-call-success.wav")            

            else:
                print(f"⚠️ 未知工具: {tool_name}")
        else:
            print("❌ 模型未调用任何工具")              
    except Exception:
        print(traceback.format_exc())
        gr.Warning("Some error happened, please try again.")
    return chatbot, history


def predict_stream(chatbot, history, audio_model, token2wav, prompt_wav, cache_dir):
    try:
        # Request speech response
        history.append({"role": "assistant", "content": "<tts_start>", "eot": False})

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "turn_on_air_conditioner",
                    "description": "开启智能座舱内空调并设置指定温度和模式",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "temperature": {"type": "integer", "description": "目标温度，16-30℃"},
                            "mode": {"type": "string", "enum": ["cool", "heat", "fan", "auto"], "description": "模式"}
                        },
                        "required": ["temperature", "mode"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "turn_on_play_music",
                    "description": "开启音乐播放器并设置播放模式",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "mode": {"type": "string", "enum": ["single_repeat", "list_loop","random"], "description": "播放模式"}
                        },
                        "required": ["mode"]
                    }
                }
            }
        ]

        stop_adding_tasks = False
        response = {"tts_content": {"tts_text": '', "tts_audio": ''}, "tool_calls": []}
        text_buffer = ""
        audio_buffer = []
        count = 0
        for line, text, audio in audio_model.stream(history, tools=tools, max_tokens=4096, repetition_penalty=1.05, top_p=0.9, temperature=0.7):
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


                if audio:
                    response["tts_content"]["tts_audio"] += line.get("tts_content", {}).get("tts_audio", '')
                    # print("语音token>>", audio, end='', flush=True)
                    tts_task_queue.put(audio)
                    audio_buffer.append(audio[0])
                    count += 1
                    if count % 100 == 0:
                        print("\n生成100个tokens", count / 100, datetime.now())                

        # Convert audio tokens to waveform and append to chat
        if audio_buffer:
            audio_buffer = [x for x in audio_buffer if x < 6561]
            audio_bytes = token2wav(audio_buffer, prompt_wav)
            audio_path = save_tmp_audio(audio_bytes, cache_dir)
            chatbot.append({"role": "assistant", "content": {"path": audio_path}})
            # history[-1] = {"role": "assistant", "tts_content": response["tts_content"]}
            history[-1] = {"role": "assistant", "content": text_buffer}            
        else:
            chatbot.append({"role": "assistant", "content": text_buffer})
            history[-1] = {"role": "assistant", "content": text_buffer}

        # === 处理工具调用 ===
        if response.get("tool_calls"):
            tool_calls = response["tool_calls"]
            tool_call = tool_calls[0]  # 假设只调一个
            tool_name = tool_call["function"]["name"]
            try:
                args = json.loads(tool_call["function"]["arguments"])  # 解析参数
            except json.JSONDecodeError as e:
                print("参数解析失败:", e)
                return

            # === ★ 核心：判断并调用本地函数 ★ ===
            if tool_name in AVAILABLE_TOOLS:
                # 执行本地函数
                tool_result = AVAILABLE_TOOLS[tool_name](**args)
                print("✅ 工具调用成功>>", AVAILABLE_TOOLS[tool_name])
                # 更新 history：先删掉原来的 assistant 请求
                # history.pop(-1)

                # 添加完整交互链
                history += [
                    # 模型发起工具调用
                    {
                        "role": "assistant",
                        "tool_calls": tool_calls
                    },
                    # 工具执行结果（模拟从设备/API 返回）
                    {
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": tool_result  # ← 这是 turn_on_air_conditioner 的返回值
                    },
                    # 模型继续生成自然语言回复
                    {
                        "role": "assistant",
                        "content": "<tts_start>",
                        "eot": False
                    }
                ]

                # 第二次调用：模型基于工具结果生成语音回复
                print("历史会话>>", history)
                final_response, final_text, final_audio = audio_model(
                    history, tools=tools, max_tokens=512,
                    repetition_penalty=1.05, top_p=0.9, temperature=0.7
                )

                if final_audio:
                    # tts_task_queue.put(final_audio)

                    for au in final_audio:
                        if au < 6561:
                            tts_task_queue.put([au])

                    final_audio = [x for x in final_audio if x < 6561]
                    audio_bytes = token2wav(final_audio, prompt_wav)
                    audio_path = save_tmp_audio(audio_bytes, cache_dir)
                    chatbot.append({"role": "assistant", "content": {"path": audio_path}})
                    # history[-1] = {"role": "assistant", "tts_content": response["tts_content"]}
                    history[-1] = {"role": "assistant", "content": final_text}
                else:
                    chatbot.append({"role": "assistant", "content": final_text})
                    history[-1] = {"role": "assistant", "content": final_text}

                print("最终回复:", final_text)
                # if final_audio:
                #     final_audio = [x for x in final_audio if x < 6561]
                #     wav_data = token2wav(final_audio, prompt_wav='assets/default_female.wav')
                #     with open('output-tool-call-success.wav', 'wb') as f:
                #         f.write(wav_data)
                #     print("✅ 音频已生成：output-tool-call-success.wav")            

            else:
                print(f"⚠️ 未知工具: {tool_name}")
        else:
            print("❌ 模型未调用任何工具")  
        stop_adding_tasks = True                        
    except Exception:
        print(traceback.format_exc())
        gr.Warning("Some error happened, please try again.")
    return chatbot, history

def _launch_demo(args, audio_model, token2wav):
    with gr.Blocks(delete_cache=(86400, 86400)) as demo:
        gr.Markdown("""<center><font size=8>Step Audio 2 vLLM Demo (Text-only Output)</center>""")
        with gr.Row():
            system_prompt = gr.Textbox(
                label="System Prompt",
                # value="你的名字叫做小跃，是由阶跃星辰公司训练出来的语音大模型。\n你情感细腻，观察能力强，擅长分析用户的内容，并作出善解人意的回复，说话的过程中时刻注意用户的感受，富有同理心，提供多样的情绪价值。\n今天是2025年8月29日，星期五\n请用默认女声与用户交流。",
                value="你的名字叫做小跃，是一个智能座舱语音助手，你可以调用座舱内工具（空调、音乐播放器等）。你很聪明，能根据用户要求调用相应的工具完成相应的任务，比如，用户说”打开空调”或者“太热了”，你将调用空调工具；再比如，用户说“打开音乐播放器”或者“我想听歌”，你将调用音乐播放器工具。\n请用默认女声温柔的与用户交流。",                
                lines=2,
            )
        chatbot = gr.Chatbot(
            elem_id="chatbot",
            min_height=800,
            type="messages",
        )
        history = gr.State([{"role": "system", "content": system_prompt.value}])
        mic = gr.Audio(type="filepath")
        text = gr.Textbox(placeholder="Enter message ...")

        with gr.Row():
            clean_btn = gr.Button("🧹 Clear History (清除历史)")
            regen_btn = gr.Button("🤔️ Regenerate (重试)")
            submit_btn = gr.Button("🚀 Submit")

        def on_submit(chatbot, history, mic, text):
            chatbot, history, error = add_message(chatbot, history, mic, text)
            if error:
                gr.Warning(error)
                return chatbot, history, None, None
            else:
                chatbot, history = predict_stream(chatbot, history, audio_model, token2wav, args.prompt_wav, args.cache_dir)
                return chatbot, history, None, None

        submit_btn.click(
            fn=on_submit,
            inputs=[chatbot, history, mic, text],
            outputs=[chatbot, history, mic, text],
            concurrency_limit=4,
            concurrency_id="gpu_queue",
        )

        clean_btn.click(
            fn=reset_state,
            inputs=[system_prompt],
            outputs=[chatbot, history],
        )

        def regenerate(chatbot, history):
            while chatbot and chatbot[-1]["role"] == "assistant":
                chatbot.pop()
            while history and history[-1]["role"] == "assistant":
                history.pop()
            return predict(chatbot, history, audio_model, token2wav, args.prompt_wav, args.cache_dir)

        regen_btn.click(
            regenerate,
            [chatbot, history],
            [chatbot, history],
            concurrency_id="gpu_queue",
        )

    # demo.queue().launch(
    #     server_port=args.server_port,
    #     server_name=args.server_name,
    # )
    demo.queue().launch(
        share=True,
        server_port=args.server_port,
        server_name=args.server_name,
        ssl_certfile="./cert.pem",
        ssl_keyfile="./key.pem",
        ssl_verify=False
    )

# === 异步任务队列 ===
tts_task_queue = queue.Queue()  # 存放待合成的文本
playback_queue = queue.Queue()  # 存放已合成的音频数据
stop_adding_tasks = False

if __name__ == "__main__":
    import os
    from argparse import ArgumentParser

    from stepaudio2vllm import StepAudio2
    from token2wav import Token2wav

    parser = ArgumentParser()
    parser.add_argument("--api-url", type=str, default="http://localhost:8000/v1/chat/completions", help="vLLM OpenAI-compatible endpoint")
    parser.add_argument("--model-name", type=str, default="step-audio-2-mini", help="Model name for vLLM serving")
    parser.add_argument("--token2wav-path", type=str, default=None, help="Path to token2wav directory (defaults to Step-Audio-2-mini/token2wav)")
    parser.add_argument("--prompt-wav", type=str, default="assets/default_female.wav", help="Prompt wave for the assistant.")
    parser.add_argument("--cache-dir", type=str, default="/tmp/stepaudio2", help="Cache directory for generated audio.")
    parser.add_argument("--server-port", type=int, default=7862, help="Demo server port.")
    parser.add_argument("--server-name", type=str, default="0.0.0.0", help="Demo server name.")
    args = parser.parse_args()

    os.environ["GRADIO_TEMP_DIR"] = args.cache_dir
    Path(args.cache_dir).mkdir(parents=True, exist_ok=True)

    prompt_wav = "assets/default_female.wav"

    audio_model = StepAudio2(args.api_url, args.model_name)
    token2wav_path = args.token2wav_path or "/home/promote/.cache/modelscope/hub/models/stepfun-ai/Step-Audio-2-mini/token2wav"
    token2wav = Token2wav(token2wav_path)

    tokens = [1493, 4299, 4218, 2049, 528, 2752, 4850, 4569, 4575, 6372, 2127, 4068, 2312, 4993, 4769, 2300, 226, 2175, 2160, 2152, 6311, 6065, 4859, 5102, 4615, 6534, 6426, 1763, 2249, 2209, 5938, 1725, 6048, 3816, 6058, 958, 63, 4460, 5914, 2379, 735, 5319, 4593, 2328, 890, 35, 751, 1483, 1484, 1483, 2112, 303, 4753, 2301, 5507, 5588, 5261, 5744, 5501, 2341, 2001, 2252, 2344, 1860, 2031, 414, 4366, 4366, 6059, 5300, 4814, 5092, 5100, 1923, 3054, 4320, 4296, 2148, 4371, 5831, 5084, 5027, 4946, 4946, 2678, 575, 575, 521, 518, 638, 1367, 2804, 3402, 4299]
    token2wav.set_stream_cache(prompt_wav)

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
                    token = tts_task_queue.get(timeout=0.5)[0]  # 假设 token 结构是 [token_id]
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
            # time.sleep(0.01)  # 避免忙等待

    # 在程序启动时添加预热逻辑
    def warmup_tts():
        warmup_tokens = [1493, 4299, 4218]  # 示例 token
        try:
            token2wav.stream(warmup_tokens, prompt_wav=prompt_wav, last_chunk=True)
            print("TTS 预热完成")
        except Exception as e:
            print(f"预热失败: {e}")

    # 启动后台线程
    threading.Thread(target=tts_worker, daemon=True).start()
    threading.Thread(target=playback_worker, daemon=True).start()

    # 在后台线程中进行预热
    def background_warmup():
        # time.sleep(2)  # 等待主要组件初始化完成
        warmup_tts()

    threading.Thread(target=background_warmup, daemon=True).start()

    _launch_demo(args, audio_model, token2wav)
