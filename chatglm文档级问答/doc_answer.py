import os
from pypdf import PdfReader
import docx
import faiss
from sentence_transformers import SentenceTransformer
def get_data(path):
    all_content = []
    files = os.listdir(path)

    for file in files:
        pa = os.path.join(path,file)
        if pa.endswith('.docx'):
            doc = docx.Document(pa)
            para = doc.paragraphs
            content = "\n".join([i.text for i in para])
            all_content.append(content)
        elif pa.endswith('.pdf'):
            content = ""
            with open(pa,"rb") as f:
                pdf_reader = PdfReader(f)

                pages_info = pdf_reader.pages
                for page_info in pages_info:
                    text = page_info.extract_text()
                    content+=text
            all_content.append(content)
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
        return self.text_str_list[I[0][0]]
if __name__ == "__main__":
    all_content = get_data('datas')
    myfaiss = DFaiss()
    myfaiss.add_content(all_content)
    while True:
        text = input('user:')

        print(myfaiss.search(text))