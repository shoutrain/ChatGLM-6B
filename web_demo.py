from transformers import AutoModel, AutoTokenizer
import gradio as gr
import mdtex2html

model_path = "/home/guxiaoqun/models/THUDM/chatglm-6b-int4"

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = (
    AutoModel.from_pretrained(model_path, trust_remote_code=True).half().cuda()
)
model = model.eval()


# 定义模型回答逻辑
def generate_response(prompt, max_length, top_p, temperature, history):
    """
    根据用户输入生成模型响应。
    :param prompt: 用户输入
    :param max_length: 最大生成长度
    :param top_p: top-p 采样参数
    :param temperature: 温度参数
    :param history: 历史对话列表
    :return: 更新后的对话记录
    """
    response, updated_history = model.chat(
        tokenizer,
        prompt,
        history=history,
        max_length=max_length,
        top_p=top_p,
        temperature=temperature,
    )
    updated_history = history + [[prompt, response]]
    return updated_history, updated_history


# Gradio 用户界面
with gr.Blocks() as demo:
    gr.Markdown("# 老顾的本地模型（RTX 2060 Super 8G）用户界面")
    gr.Markdown(
        "可以通过滑块调整参数，如 `max_length`、`top_p` 和 `temperature`。"
    )

    with gr.Row():
        with gr.Column(scale=3):
            prompt_input = gr.Textbox(
                label="输入问题", placeholder="请输入您的问题...", lines=4
            )
            max_length_slider = gr.Slider(
                minimum=32,
                maximum=2048,
                value=512,
                step=1,
                label="最大生成长度 (max_length)",
            )
            top_p_slider = gr.Slider(
                minimum=0.1,
                maximum=1.0,
                value=0.9,
                step=0.1,
                label="Top-p 采样 (top_p)",
            )
            temperature_slider = gr.Slider(
                minimum=0.1,
                maximum=1.0,
                value=0.7,
                step=0.1,
                label="温度 (temperature)",
            )
            submit_button = gr.Button("生成回答")
        with gr.Column(scale=5):
            chat_output = gr.Chatbot(label="对话记录")
            clear_button = gr.Button("清空会话")

    # 历史会话状态
    state = gr.State([])

    # 按钮点击逻辑
    submit_button.click(
        fn=generate_response,
        inputs=[
            prompt_input,
            max_length_slider,
            top_p_slider,
            temperature_slider,
            state,
        ],
        outputs=[chat_output, state],
    )
    clear_button.click(
        fn=lambda: ([], []),  # 清空历史记录
        inputs=[],
        outputs=[chat_output, state],
    )

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", share=False, inbrowser=True)
