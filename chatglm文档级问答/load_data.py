import os
from pypdf import PdfReader
import docx
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
            content = []
            with open(pa,"rb") as f:
                pdf_reader = PdfReader(f)

                pages_info = pdf_reader.pages
                for page_info in pages_info:
                    text = page_info.extract_text()
                    content+=text
            content = "".join(content)
            all_content.append(content)
        elif pa.endswith('.txt'):
            with open(pa,'r',encoding='utf8') as f:
                all_data = f.read()
            all_content.append(all_data)
    return all_content
if __name__ == "__main__":
    all_content = get_data('datas')
    print()