import traceback
from langchain.document_loaders import PyPDFLoader, UnstructuredPDFLoader

data_location = "/home/azureuser/codebase/zodiac-2007.pdf"


def read_pdf_file(file_location, url=False):
    if url:
        print("Reading File from Cloud Storage URL is not yet Implemented")
        return "Reading File from Cloud Storage URL is not yet Implemented", False
    else:

        if file_location:
            try:
                loader = PyPDFLoader(file_location)
                return loader, True
            except Exception as e:
                print(traceback.format_exc())
                return "Exception occured at langchain pdf file reader", False
        else:
            return "File location is empty", False

def return_pdf_pages(loader):
    try:
        return loader.load_and_split(), True
    except Exception as e:
        print(traceback.format_exc())
        return "Exception occured at langchain pdf data splitter", False

def return_all_script_lines_combined(pages):
    try:
        return "\n".join([x.page_content for x in pages]), True
    except Exception as e:
        print(traceback.format_exc())
        return "Exception occured at langchain page content", False


def extract_individual_scene_list_from_script(script):
    try:
        stripped_scene_list = script.split("\n") #### Separating script into lines, to extract individual scenes which comes between INT/EXT to next INT/EXT
        scene_info_list = []
        scene_index_list = [(i, scene) for i,scene in enumerate(stripped_scene_list) if  scene.startswith("EXT") or scene.startswith("INT")]
        for i,scene_index in enumerate(scene_index_list):
            complete_scene_info_from_stripped_text = ""
            each_scene_start_index, each_scene_stop_index = 0,0
            if i< len(scene_index_list)-1:

                each_scene_start_index, each_scene_stop_index = scene_index_list[i][0], scene_index_list[i+1][0]
                complete_scene_info_from_stripped_text = stripped_scene_list[each_scene_start_index : each_scene_stop_index]
                complete_scene_info_from_stripped_text = [x for x in complete_scene_info_from_stripped_text if x != ""]
                scene_info_list.append((complete_scene_info_from_stripped_text[0].replace("EXT.", "").replace("INT.", "").replace("EXT./INT.","").strip(), " ".join(complete_scene_info_from_stripped_text[1:])))

            else:
                each_scene_start_index = scene_index_list[i][0]
                complete_scene_info_from_stripped_text = stripped_scene_list[each_scene_start_index : -1]
                complete_scene_info_from_stripped_text = [x for x in complete_scene_info_from_stripped_text if x != ""]

                scene_info_list.append(( complete_scene_info_from_stripped_text[0].replace("EXT.", "").replace("INT.", "").strip(), " ".join(complete_scene_info_from_stripped_text[1:])))
        return scene_info_list, True
    except Exception as e:
        print(traceback.format_exc())
        return "Exception occured at individual scene extraction", False


def generate_storyboard_prompts_from_scene(scene):

    return None, False
def generate_shot_division_from_scene(scene):
    return None, False
def generate_scene_summary(scene):
    return None, False

scene_list = []
footnote_for_storyboard_creation_in_prompts = "\n Create storyboard frames for the above scene"

def main(file_location):
    global scene_list
    loader, check = read_pdf_file(file_location)
    if check:
        pages, check = return_pdf_pages(loader)
        if check:
            script,check = return_all_script_lines_combined(pages)
            if check:
                scene_list, check = extract_individual_scene_list_from_script(script)
                if check:
                    for scene in scene_list:
                        storyboard_from_llm, check = ""


    

# model = AutoModel.from_pretrained("THUDM/chatglm-6b-int4", trust_remote_code=True).to("cpu", dtype=float)


"""
INT. SAN FRANCISCO CHRONICLE OFFICE - DAY
A bustling newsroom filled with REPORTERS and EDITORS typing away at their desks. Loud phones ringing and the clatter of typewriters fill the air.
We follow ROBERT GRAYSMITH, a quiet and unassuming CARTOONIST, as he walks through the chaotic room. He carries a stack of papers and nervously glances around.
INT. ROBERT GRAYSMITH'S OFFICE - DAY
Robert enters his small office cluttered with DRAWINGS and reference materials. He places the stack of papers on his desk.
"""


prompt = """
INT. SAN FRANCISCO CHRONICLE OFFICE - DAY
A bustling newsroom filled with REPORTERS and EDITORS typing away at their desks. Loud phones ringing and the clatter of typewriters fill the air.
We follow ROBERT GRAYSMITH, a quiet and unassuming CARTOONIST, as he walks through the chaotic room. He carries a stack of papers and nervously glances around.
INT. ROBERT GRAYSMITH'S OFFICE - DAY
Robert enters his small office cluttered with DRAWINGS and reference materials. He places the stack of papers on his desk.
For the above scene, can you give me shot divisions?
"""


import pdfplumber

with pdfplumber.open("zodiac-2007.pdf") as pdf:
    first_page = pdf.pages[0]
    print(first_page.chars[0])


import fitz

def get_text(filepath: str) -> str:
    with fitz.open(filepath) as doc:
        text = ""
        for page in doc:
            text += page.getText().strip()
            break
        return text
filepath = "zodiac-2007.pdf"

with open("zodiac-2007.pdf", "rb") as f:
    pdf = pdftotext.PDF(f)

# All pages
for text in pdf:
    print(text)
    break
