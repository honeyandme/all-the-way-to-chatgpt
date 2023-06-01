import os
from pypdf import PdfReader
import docx
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import gradio as gr
import shutil
def get_data(path):
    all_content = []
    files = os.listdir(path)

    for file in files:
        pa = os.path.join(path,file)
        if pa.endswith('.docx'):
            doc = docx.Document(pa)
            para = doc.paragraphs
            content = [i.text for i in para]
            text = ""
            for con in content:

                if(len(con)<=1):
                    continue
                text += con
                if len(text)>=100:
                    all_content.append(text)
                    text = ""
            if len(text)>1:
                all_content.append(text)

            # all_content.append("\n".join(content))
        elif pa.endswith('.pdf'):
            # content = ""
            with open(pa,"rb") as f:
                pdf_reader = PdfReader(f)

                pages_info = pdf_reader.pages
                for page_info in pages_info:
                    text = page_info.extract_text()
                    # content+=text
                    all_content.append(text)
        elif pa.endswith('.txt'):
            with open(pa,'r',encoding='utf8') as f:
                all_data = f.read()
            all_content.append(all_data)
    return all_content

class DFaiss:
    def __init__(self):
        self.sentence_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2',cache_folder='sen_model')
        self.index = faiss.IndexFlatL2(384)
        self.text_str_list = []
    def add_content(self,strlist):
        text_emb = self.get_text_emb(strlist)
        self.text_str_list.extend(strlist)
        self.index.add(text_emb)
    def get_text_emb(self,strlist):
        return self.sentence_model.encode(strlist)
    def search(self,text):
        text_emb = self.get_text_emb([text])
        D,I = self.index.search(text_emb,3)
        if D[0][0]<15:
            return self.text_str_list[I[0][0]]
        else:
            return ""

class prompt_robot():
    def __init__(self):
        self.myfaiss = DFaiss()
        self.tokenizer = AutoTokenizer.from_pretrained("../THUDM/chatglm-6b", trust_remote_code=True)
        self.model = AutoModel.from_pretrained("../THUDM/chatglm-6b", trust_remote_code=True).half().cuda()
        self.model.eval()
        self.history = []
    def ask(self,query):
        search_result = self.myfaiss.search(query)
        if len(search_result)==0:
            prompt = query
        else:
            prompt=f'请根据以下内容回答问题。内容是"{search_result}",问题是"{query}"。'
        # print(f'the prompt is {prompt}')
        response, self.history = self.model.chat(self.tokenizer, prompt, history=self.history)
        self.history = self.history[-10:]
        return response
    def add_content(self,strlist):
        self.myfaiss.add_content(strlist)

def load_file(files):
    if os.path.exists('temp'):
        shutil.rmtree('temp')
    os.mkdir('temp')

    for file in files:
        n = os.path.basename(file.orig_name)
        p = os.path.join('temp',n)
        shutil.move(file.name,p)
        print('ok')
    return [[None,'文件加载成功～']]


def ans(query, history):
    global robot

    return history + [[query, robot.ask(query)]]
def ans_stream(query,history):
    global robot
    search_result = robot.myfaiss.search(query)
    result = history+[[query,""]]
    if len(search_result)==0:
        prompt = query
    else:
        prompt=f'请根据以下内容回答问题。内容是"{search_result}",问题是"{query}"。'
    for res,his in robot.model.stream_chat(robot.tokenizer, prompt, history=[]):
        result[-1] = [query,res]
        yield result
if __name__ == "__main__":
    robot = prompt_robot()
    with gr.Blocks() as Robot:
        with gr.Row():
            with gr.Column(scale=3):
                chatbot=gr.Chatbot(
                    [[None,'这里是zwk机器人']]
                ).style(height=600)
                query=gr.Textbox(placeholder='请输出问题，回车发送:')
            query.submit(ans_stream,inputs=[query,chatbot],outputs=chatbot,show_progress=True)
            with gr.Column(scale=1):
                file = gr.File(file_count="multiple")
                button = gr.Button('加载文件')
            button.click(load_file,inputs=[file,chatbot],outputs=chatbot)
    Robot.queue().launch(server_name='127.0.0.1',server_port=6006,share=True)
    # all_content = get_data('datas')
    # robot = prompt_robot()
    # robot.add_content(all_content)
    # while True:
    #     print('----------------------------------')
    #     text = input('user:')
    #     print(f'robot:{robot.ask(text)}')