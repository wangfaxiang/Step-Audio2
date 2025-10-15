from stepaudio2vllm import StepAudio2
from token2wav import Token2wav
import json
import time
# ASR
def asr_test(model):
    messages = [
        {"role": "system", "content": "请记录下你所听到的语音内容。"},
        {"role": "human", "content": [{"type": "audio", "audio": "assets/give_me_a_brief_introduction_to_the_great_wall.wav"}]},
        {"role": "assistant", "content": None}
    ]
    _, text, _ = model(messages, max_tokens=1024, temperature=0)
    print(text)


# S2TT（support: en,zh,ja）
def s2tt_test(model):
    messages = [
        {"role": "system", "content":"请仔细聆听这段语音，然后将其内容翻译成中文。"},
        # {"role": "system", "content":"Please listen carefully to this audio and then translate its content into Chinese."},
        {"role": "human", "content": [{"type": "audio", "audio": "assets/give_me_a_brief_introduction_to_the_great_wall.wav"}]},
        {"role": "assistant", "content": None}
    ]
    _, text, _ = model(messages, max_tokens=1024, temperature=0.1)
    print(text)


# audio caption
def audio_caption_test(model):
    messages = [
        {"role": "system", "content":"Please briefly explain the important events involved in this audio clip."},
        {"role": "human", "content": [{"type": "audio", "audio": "assets/music_playing_followed_by_a_woman_speaking.wav"}]},
        {"role": "assistant", "content": None}
    ]
    _, text, _ = model(messages, max_tokens=1024, temperature=0.1)
    print(text)


# S2ST（support: en,zh）
def s2st_test(model, token2wav):
    messages = [
        {"role": "system", "content":"请仔细聆听这段语音，然后将其内容翻译成中文并用语音播报。"},
        # {"role": "system", "content":"Please listen carefully to this audio and then translate its content into Chinese speech."},
        {"role": "human", "content": [{"type": "audio", "audio": "assets/give_me_a_brief_introduction_to_the_great_wall.wav"}]},
        {"role": "assistant", "content": "<tts_start>", "eot": False},
    ]
    _, text, audio = model(messages, max_tokens=2048, temperature=0.7)
    print(text)
    if audio:
        audio = [x for x in audio if x < 6561]
        audio = token2wav(audio, prompt_wav='assets/default_female.wav')
        with open('output-s2st.wav', 'wb') as f:
            f.write(audio)


# multi turn speech-to-text conversation
def multi_turn_aqta_test(model):
    history = [{"role": "system", "content": "You are a helpful assistant."}]
    for round_idx, inp_audio in enumerate([
        "assets/multi-turn-round1-听说荡口古镇从下个月开始取消门票了，你知道这事吗。.wav",
        "assets/multi-turn-round2-新闻说九月十九号就免费开放了。好像整个古镇都升级改造了，现在变成开放式街区了。.wav"
    ]):
        print("round: ", round_idx)
        history.append({"role": "human", "content": [{"type": "audio", "audio": inp_audio}]})
        history.append({"role": "assistant", "content": None})
        _, text, _ = model(history, max_tokens=1024, temperature=0.5)
        print(text)
        history.pop(-1)
        history.append({"role": "assistant", "content": text})


# multi turn speech-to-speech conversation
def multi_turn_aqaa_test(model, token2wav):
    history = [{"role": "system", "content": "You are a helpful assistant."}]
    for round_idx, inp_audio in enumerate([
        "assets/multi-turn-round1-听说荡口古镇从下个月开始取消门票了，你知道这事吗。.wav",
        "assets/multi-turn-round2-新闻说九月十九号就免费开放了。好像整个古镇都升级改造了，现在变成开放式街区了。.wav"
    ]):
        print("round: ", round_idx)
        history.append({"role": "human", "content": [{"type": "audio", "audio": inp_audio}]})
        history.append({"role": "assistant", "content": "<tts_start>", "eot": False})
        response, text, audio = model(history, max_tokens=2048, temperature=0.7)
        print(text)
        if audio:
            audio = [x for x in audio if x < 6561]
            audio = token2wav(audio, prompt_wav='assets/default_female.wav')
            with open(f'output-round-{round_idx}.wav', 'wb') as f:
                f.write(audio)
        history.pop(-1)
        history.append({"role": "assistant", "tts_content": response.get("tts_content", {})})

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
    
    return f"已为您开启空调，设定温度为{temperature}度，运行模式为{mode}。"

# === 工具映射表：把名字映射到实际函数 ===
AVAILABLE_TOOLS = {
    "turn_on_air_conditioner": turn_on_air_conditioner,
}

# Tool call & Web search
def tool_call_test(model, token2wav):
    history = [
        # {"role": "system", "content": "你的名字叫做小跃，是由阶跃星辰公司训练出来的语音大模型。\n你具备调用工具解决问题的能力，你需要根据用户的需求和上下文情景，自主选择是否调用系统提供的工具来协助用户。\n你情感细腻，观察能力强，擅长分析用户的内容，并作出善解人意的回复，说话的过程中时刻注意用户的感受，富有同理心，提供多样的情绪价值。\n今天是2025年8月28日，星期四\n请用默认女声与用户交流"},
        {"role": "system", "content": "你的名字叫做小跃，是一个智能座舱语音助手，具备调用工具（空调、车灯、雨刷等）解决问题的能力。当用户提出与车内环境调节相关的需求时，必须立即调用相应的控制工具，比如，用户说”太热了”，请调用空调控制工具，设定制冷模式和合适的温度。\n请用默认女声温柔的与用户交流"},        # {"role": "human", "content": [{"type": "audio", "audio": "assets/帮我查一下今天上证指数的开盘价是多少.wav"}]},        # {"role": "tool_json_schemas", "content": '[{"type": "function", "function": {"name": "search", "description": "搜索工具", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "搜索关键词"}}, "required": ["query"], "additionalProperties": false}}}]'},        
        # {"role": "human", "content": [{"type": "audio", "audio": "assets/帮我查一下今天上证指数的开盘价是多少.wav"}]},        
        {"role": "human", "content": [{"type": "text", "text": "车里好热啊，开下空调吧，调到24度制冷。"}]},
        {"role": "assistant", "content": "<tts_start>", "eot": False},
    ]

    # tools = [{"type": "function", "function": {"name": "search", "description": "搜索工具", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "搜索关键词"}}, "required": ["query"], "additionalProperties": False}}}]
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
                "name": "search",
                "description": "用于查询实时信息，如天气、指数、新闻等",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "搜索关键词"}
                    },
                    "required": ["query"]
                }
            }
        }
    ]

    start_time = time.time()  # ✅ 函数开始时间戳
    print(f"【开始执行】tool_call_test 启动时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")

    response, text, audio = model(history, tools=tools, max_tokens=4096, repetition_penalty=1.05, top_p=0.9, temperature=0.7)
    print("di yi ci hui fu>>", text)
    print(response["tool_calls"])
    if audio:
        audio = [x for x in audio if x < 6561]
        audio = token2wav(audio, prompt_wav='assets/default_female.wav')
        with open('output-tool-call-1.wav', 'wb') as f:
            f.write(audio)

    end_time1 = time.time()  # ✅ 函数结束时间戳
    duration1 = end_time1 - start_time
    print(f"【执行完成】tool_call_test 结束时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time1))}")
    print(f"⏱️  总耗时: {duration1:.2f} 秒")

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
            history.pop(-1)

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
            final_response, final_text, final_audio = model(
                history, tools=tools, max_tokens=512,
                repetition_penalty=1.05, top_p=0.9, temperature=0.7
            )

            print("最终回复:", final_text)
            if final_audio:
                final_audio = [x for x in final_audio if x < 6561]
                wav_data = token2wav(final_audio, prompt_wav='assets/default_female.wav')
                with open('output-tool-call-success.wav', 'wb') as f:
                    f.write(wav_data)
                print("✅ 音频已生成：output-tool-call-success.wav")
                end_time2 = time.time()  # ✅ 函数结束时间戳
                duration2 = end_time2 - end_time1
                print(f"【执行完成】tool_call_test 结束时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time2))}")
                print(f"⏱️  总耗时: {duration2:.2f} 秒")            

        else:
            print(f"⚠️ 未知工具: {tool_name}")
    else:
        print("❌ 模型未调用任何工具")       

    # history.pop(-1)
    # with open('assets/search_result.txt') as f:
    #     search_result = f.read().strip()
    # history += [
    #     {"role": "assistant", "tts_content": response["tts_content"], "tool_calls": response["tool_calls"]},
    #     {"role": "input", "tool_call_id": response["tool_calls"][0]["id"], "content": [{"type": "text", "text": search_result}, {"type": "text", "text": '\n\n\n请用口语化形式总结检索结果，简短地回答用户的问题。'}]},
    #     {"role": "assistant", "content": "<tts_start>", "eot": False},
    # ]
    # response, text, audio = model(history, tools=tools, max_tokens=4096, repetition_penalty=1.05, top_p=0.9, temperature=0.7)
    # print(text)
    # if audio:
    #     audio = [x for x in audio if x < 6561]
    #     audio = token2wav(audio, prompt_wav='assets/default_female.wav')
    #     with open('output-tool-call-2.wav', 'wb') as f:
    #         f.write(audio)


# Paralinguistic information understanding
def paralinguistic_test(model, token2wav):
    messages = [
        {"role": "system", "content":"请用语音与我交流。"},
        {"role": "human", "content": [{"type": "audio", "audio": "assets/paralinguistic_information_understanding.wav"}]},
        {"role": "assistant", "content": "<tts_start>", "eot": False},
    ]
    _, text, audio = model(messages, max_tokens=2048, temperature=0.7)
    print(text)
    if audio:
        audio = [x for x in audio if x < 6561]
        audio = token2wav(audio, prompt_wav='assets/default_female.wav')
        with open('output-paralinguistic.wav', 'wb') as f:
            f.write(audio)


# Audio understanding
def mmau_test(model):
    messages = [
        {"role": "system", "content": "You are an expert in audio analysis, please analyze the audio content and answer the questions accurately."},
        {"role": "human", "content": [{"type": "audio", "audio": "assets/mmau_test.wav"}]},
                                      {"type": "text", "text": f"Which of the following best describes the male vocal in the audio? Please choose the answer from the following options: [Soft and melodic, Aggressive and talking, High-pitched and singing, Whispering] Output the final answer in <RESPONSE> </RESPONSE>."},
        {"role": "assistant", "content": None}
    ]
    _, text, _ = model(messages, max_tokens=1024, best_of=2, use_beam_search=True)
    print(text)


# Universal audio caption
def uac_test(model):
    messages = [
        {"role": "system", "content": "你是一位经验丰富的音频分析专家，擅长对各种语音音频进行深入细致的分析。你的任务不仅仅是将音频内容准确转写为文字，还要对说话人的声音特征（如性别、年龄、情绪状态）、背景声音、环境信息以及可能涉及的事件进行全面描述。请以专业、客观的视角，详细、准确地完成每一次分析和转写。"},
        {"role": "human", "content": [{"type": "audio", "audio": "assets/music_playing_followed_by_a_woman_speaking.wav"}]},
        {"role": "assistant", "content": None}
    ]
    _, text, _ = model(messages, max_tokens=1024, temperature=0.5, top_p=0.9)
    print(text)

if __name__ == '__main__':
    api_url = "http://localhost:8999/v1/chat/completions"
    model_name = "step-audio-2-mini"

    model = StepAudio2(api_url, model_name)
    token2wav = Token2wav('/home/promote/.cache/modelscope/hub/models/stepfun-ai/Step-Audio-2-mini/token2wav')

    # asr_test(model)
    # s2tt_test(model)
    # audio_caption_test(model)
    # s2st_test(model, token2wav)
    # multi_turn_aqta_test(model)
    # multi_turn_aqaa_test(model, token2wav)
    tool_call_test(model, token2wav)
    # paralinguistic_test(model, token2wav)
    # mmau_test(model)
    # uac_test(model)
