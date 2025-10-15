from langchain_core.documents import Document

doc=Document(
    page_content="this is main text content I am using to create RAG",
    metadata={
        "source":"example.txt",
         "pages":1,
         "author":"Anusha Upadya",
         "date_created":"2025-10-15"
    }
)

# print(doc)

# from langchain.document_loaders import TextLoader
from langchain_community.document_loaders import TextLoader

loader=TextLoader("data/text_files/supervised.txt",encoding="utf-8")

document=loader.load()
# print(document)

from langchain_community.document_loaders import DirectoryLoader

## load all the text files from the directory
dir_loader=DirectoryLoader(
"data/text_files",
glob="*.txt", ## Pattern to match files
loader_cls= TextLoader, ##loader class to use
loader_kwargs={'encoding': 'utf-8'},
show_progress=False
)

document=dir_loader.load()
# print(document)

from langchain_community.document_loaders import PyMuPDFLoader

pdf_dir_loader=DirectoryLoader(
"data/pdf_files",
glob="*.pdf", ## Pattern to match files
loader_cls= PyMuPDFLoader, ##loader class to use
show_progress=False
)

document=pdf_dir_loader.load()
# print(document)
