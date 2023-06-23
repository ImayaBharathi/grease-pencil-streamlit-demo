##### Default Packages
import os
import re
import json
import base64 
import traceback
from io import BytesIO
from typing import Union
from contextlib import asynccontextmanager

##### FastAPI & Pydantic
from fastapi import File
from fastapi import Body
from fastapi import FastAPI
from fastapi import Response
from fastapi import UploadFile
from pydantic import BaseModel
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware


##### Huggingface Libraries and Pytorch
import torch
from transformers import AutoModel, AutoTokenizer
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

##### External Libraries
import pdftotext
from PIL import Image


##### Setting Pytorch MAX allocated GPU Size
torch.cuda.empty_cache()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

class InputGenerateStoryBoards(BaseModel):
    scene : list

class ResponseModel(BaseModel):
    status_code: str
    msg: str
    data: str
    isDataPresent: bool


def load_llm_model():
    model_glm = "LLM"
    tokenizer_glm = "tokenizer"
    tokenizer_glm = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
    model_glm = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True).half().cuda()
    model_glm = model_glm.eval()
    return model_glm, tokenizer_glm

def load_imagegen_model():
    pipe = "Imagen"
    model_id = "darkstorm2150/Protogen_x3.4_Official_Release"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")
    return pipe

##### Global Imports for data and functions
ai_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    ai_models["llm"], ai_models["llm_tokenizer"] = load_llm_model()
    ai_models["imagen"] = load_imagegen_model()
    yield
    # Clean up the ML models and release the resources
    ai_models.clear()

app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, 
    allow_credentials=True, 
    allow_origins=["*"], 
    allow_methods=["*"], 
    allow_headers=["*"]
)


@app.get("/")
def read_root():
    return {"msg": "Server is Running"}

def read_script_file_contents(pdfFile, file_type: str):
    try:
        if file_type == "pdf":
            pdf = pdftotext.PDF(pdfFile.file)
            whole_script = ""
            for i,text in enumerate(pdf):
                if i==2:
                    print(text)
                whole_script+="\n"+text

            return whole_script, True
        else:
            return "Exception occured at script file reading", False    
    except Exception as e:
        print(traceback.format_exc())
        return "Exception occured at script file reading function", False

def read_script_file_stored_in_server(file_name: str):
    try:
        with open(file_name, "r") as f:
            script = f.read()

        return script, True
    except Exception as e:
        print(traceback.format_exc())
        return "Exception occured at script file reading from server function", False
    
def extract_individual_scene_list_from_script(script: str):
    try:
        stripped_scene_list = script.split("\n") #### Separating script into lines, to extract individual scenes which comes between INT/EXT to next INT/EXT
        
        scene_info_list = []
        scene_index_list = [(i, scene) for i,scene in enumerate(stripped_scene_list) if "EXT" in scene or "INT" in scene]
        for i,scene_index in enumerate(scene_index_list):
            complete_scene_info_from_stripped_text = ""
            each_scene_start_index, each_scene_stop_index = 0,0
            if i< len(scene_index_list)-1:
                each_scene_start_index, each_scene_stop_index = scene_index_list[i][0], scene_index_list[i+1][0]
                complete_scene_info_from_stripped_text = stripped_scene_list[each_scene_start_index : each_scene_stop_index]
                complete_scene_info_from_stripped_text = [x for x in complete_scene_info_from_stripped_text if x != ""]
                ### scene_info_list.append((complete_scene_info_from_stripped_text[0].replace("EXT.", "").replace("INT.", "").replace("EXT./INT.","").strip(), " ".join(complete_scene_info_from_stripped_text[1:])))
                ### Adding INT and EXT to scene starting
                scene_info_list.append((complete_scene_info_from_stripped_text[0], "\n".join(complete_scene_info_from_stripped_text[1:])))
            else:
                each_scene_start_index = scene_index_list[i][0]
                complete_scene_info_from_stripped_text = stripped_scene_list[each_scene_start_index : -1]
                complete_scene_info_from_stripped_text = [x for x in complete_scene_info_from_stripped_text if x != ""]
                ###scene_info_list.append(( complete_scene_info_from_stripped_text[0].replace("EXT.", "").replace("INT.", "").strip(), " ".join(complete_scene_info_from_stripped_text[1:])))
                ### Adding INT and EXT to scene starting
                scene_info_list.append(( complete_scene_info_from_stripped_text[0], "\n".join(complete_scene_info_from_stripped_text[1:])))
        return scene_info_list, True
    except Exception as e:
        print(traceback.format_exc())
        return "Exception occured at individual scene extraction", False
    
def create_prompt(scene: str, type: str):
    if type == "storyboard":
        prompt = f"""{scene} 
        - Create storyboard frames for the above scene"""
        return prompt
    elif type == "image":
        prompt = f"""{scene} 
        -Draw in pencil sketch"""
        return prompt

def generate_storyboard_descriptions(prompt:str):
    try:
        response, history = ai_models["llm"].chat(ai_models["llm_tokenizer"], prompt, history=[])
        return response, True
    except Exception as e:
        print(traceback.format_exc())
        return "Exception occured at generating storyboard descriptions", False

def has_numbers(sentence: str):
    check = [char for char in sentence if char.isdigit()]
    if check:
        return True
    else:
        return False

def parse_llm_output(model_response: str):
    try:
        if "\n" in model_response:
            sentences= model_response.split("\n")
            sentences_with_numbers = [sent for sent in sentences if has_numbers(sent)]
            if sentences_with_numbers:
                cleaned_sentences  = [re.sub("\d\.", "",sent).strip() for sent in sentences_with_numbers]
                return cleaned_sentences, True
            else:
                print(model_response)
                print("AI Response format has changed,the model response doesnt have number based ordering, check the logs")
                return "AI Response format has changed,the model response doesnt have number based ordering, check the logs", False    
        else:
            print(model_response)
            print("AI Response format has changed,there is no '\n' present in the model response, check the logs")
            return "AI Response format has changed,there is no '\n' present in the model response, check the logs", False
    except Exception as e:
        print(traceback.format_exc())
        return "Exception occured at parsing LLM Output part", False

def generate_images_for_description(prompt: str):
    try:
        image = ai_models["imagen"](prompt, num_inference_steps=35).images[0]
        return image, True
    except Exception as e:
        print(traceback.format_exc())
        return "Exception occured at Image Generation part", False

@app.post("/script_file_upload", response_model=ResponseModel, status_code=201, description="Reads a Script (currently PDF format) ", tags=["Script_File_Operations"])
async def read_item(pdfFile: UploadFile = File(...), file_type: str="pdf", file_name: str="give your script name with no spaces"):
    try:
        script, check = read_script_file_contents(pdfFile, file_type)
        if file_name.strip():
            file_name = str(file_name)+".txt"
        else:
            file_name= str(pdfFile.filename).split('.')[0]+".txt"
        print(file_name)
        if check:
            with open(f"script_files/{file_name}", "w") as f:
                f.write(script)
            return {"status_code": "201", "msg":"Script file saved successfully", "data":json.dumps({"filename": file_name}), "isDataPresent":True}
    except Exception as e:
        print(traceback.format_exc())
        return {"status_code":503, "msg": "Backend Server is currently not running", "data":json.dumps({}), "isDataPresent":False}

@app.post("/get_scenes_from_uploaded_file", response_model=ResponseModel, status_code=201, description="Gives Scenes for a given script file name", tags=["Script_File_Operations"])
async def get_scenes(file_name: str):
    if file_name.strip():
        file_name = str(file_name)+".txt"
        file_full_path = f"script_files/{file_name}"
        script, check = read_script_file_stored_in_server(file_full_path)
        if check:
            scene_list, check = extract_individual_scene_list_from_script(script)
            if check:
                msg = f"Scene Extracted for a given script file {file_name}"
                return {"status_code": "201", "msg":msg, "data":json.dumps({"scene_list": scene_list}), "isDataPresent":True}   
            else:
                print("Exception occured at Individual Scene Extraction part")
                return {"status_code":503, "msg": "Internal Server Error", "data":json.dumps({}), "isDataPresent":False}    
        else:
            print("Exception occured at text file reading")
            return {"status_code":503, "msg": "Internal Server Error", "data":json.dumps({}), "isDataPresent":False}    
    else:
        print("Filename is empty")
        return {"status_code":503, "msg": "Filename is empty, Filename is a must", "data":json.dumps({}), "isDataPresent":False}

@app.post("/get_storyboard_prompts_from_scene", response_model=ResponseModel, status_code=201, description="Scene to Storyboard Descriptions", tags=["AI Models"])
async def get_storyboard_descriptions(scene_data: str= Body(..., media_type='text/plain')):
    if scene_data.strip():
        print(scene_data)
        print
        prompt = create_prompt(scene_data, type = "storyboard")
        model_response, check = generate_storyboard_descriptions(prompt)####### Works starts from here, also add validation to model response, the response should be a list of descriptions
        if check:
            storyboard_frames, check = parse_llm_output(model_response)
            if check:
                return {"status_code": "201", "msg":"Successfully created storyboard descriptions", "data":json.dumps({"storyboard_frames": storyboard_frames}), "isDataPresent":True}   
            else:
                print("Exception occured at llm generation part")
                print(storyboard_frames)
                return {"status_code": "503", "msg":storyboard_frames, "data":json.dumps({}), "isDataPresent":False}
        else:
            print("Exception occured at llm generation part")
            print(storyboard_frames)
            return {"status_code": "503", "msg":storyboard_frames, "data":json.dumps({}), "isDataPresent":False}
        # model_response, check = generate_storyboard_descriptions(prompt) 
    else:
        print("Scene is empty")
        return {"status_code":503, "msg": "Scene is empty, Scene is must", "data":json.dumps({}), "isDataPresent":False}

@app.post("/get_images_for_storyboard_description", status_code=201, description="Generate Storyboard Images from One Liners", tags=["AI Models"])
async def generate_storyboards(storyboard_description: str):
    if storyboard_description.strip():
        prompt = create_prompt(storyboard_description.strip(),type = "image")
        model_response, check = generate_images_for_description(prompt)
        if check:
            model_response.save("testimage.png") # save the image
            # # return the image
            # buffer = BytesIO()
            # model_response.save(buffer, format="PNG")
            # imgstr = base64.b64encode(buffer.getvalue())
            return FileResponse("testimage.png")
        else:
            print(model_response)
            return {"status_code":503, "msg": model_response, "data":json.dumps({}), "isDataPresent":False}    
    else:
        print("Storyboard description is empty")
        return {"status_code":503, "msg": "Storyboard description is empty, Description is must", "data":json.dumps({}), "isDataPresent":False}