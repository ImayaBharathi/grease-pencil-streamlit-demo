import gradio as gr
import json
import requests
import os
from text_generation import Client, InferenceAPIClient

# #Load pre-trained model and tokenizer - for THUDM model
from transformers import AutoModel, AutoTokenizer
tokenizer_glm = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
model_glm = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True).half().cuda()
model_glm = model_glm.eval()

# #Load pre-trained model and tokenizer for Chinese to English translator --- chinese translator not needed
# from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
# model_chtoen = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
# tokenizer_chtoen = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")


    
# #Define function to generate model predictions and update the history
def predict_glm_stream(input, top_p, temperature, history=[]): 
    history = list(map(tuple, history))
    for response, updates in model_glm.stream_chat(tokenizer_glm, input, history, top_p=top_p, temperature=temperature):   
        yield updates 
    
def reset_textbox():
    return gr.update(value="")

title = """<h1 align="center"> 🚀Grease Pencil AI</h1>"""
description = """<br>
This is an internal deployment of Grease Pencil's LLM for testing the script integration
"""
theme = gr.themes.Default(#color contructors
                          primary_hue="violet", 
                          secondary_hue="indigo",
                          neutral_hue="purple").set(slider_color="#800080")
                           
with gr.Blocks(css="""#col_container {margin-left: auto; margin-right: auto;}
                #chatglm {height: 520px; overflow: auto;} """, theme=theme ) as demo:
    gr.HTML(title)
    with gr.Tab("LLM Model"):
        with gr.Column(): #(scale=10):
            with gr.Box():
                with gr.Row():
                    with gr.Column(scale=8):
                        inputs = gr.Textbox(placeholder="Hi there!", label="Type an input and press Enter ⤵️ " )
                    with gr.Column(scale=1):
                        b1 = gr.Button('🏃Run', elem_id = 'run').style(full_width=True)
                    with gr.Column(scale=1):
                        b2 = gr.Button('🔄Clear the Chatbot!', elem_id = 'clear').style(full_width=True)
                        state_glm = gr.State([])

            with gr.Box():
                chatbot_glm = gr.Chatbot(elem_id="chatglm", label='Grease Pencil AI')
            
            with gr.Accordion(label="Parameters to Play With", open=False):
                gr.HTML("Parameters to Play With", visible=True)
                top_p = gr.Slider(minimum=-0, maximum=1.0,value=1, step=0.05,interactive=True, label="Top-p", visible=True)
                temperature = gr.Slider(minimum=-0, maximum=5.0, value=1, step=0.1, interactive=True, label="Temperature", visible=True)

        inputs.submit( predict_glm_stream,
                    [inputs, top_p, temperature, chatbot_glm ],
                    [chatbot_glm],)
        inputs.submit(reset_textbox, [], [inputs])

        b1.click( predict_glm_stream,
                    [inputs, top_p, temperature, chatbot_glm ],
                    [chatbot_glm],)
        b1.click(reset_textbox, [], [inputs])

        b2.click(lambda: None, None, chatbot_glm, queue=False)
    with gr.Tab("ImageGen Model"):
        gr.Interface.load("models/darkstorm2150/Protogen_Infinity_Official_Release")

    gr.Markdown(description)
    demo.queue(concurrency_count=16).launch(height= 800, debug=True, server_name="0.0.0.0")