import streamlit as st
import os
import torch
import nltk
import urllib.request
from models.model_builder import Summarizer
from newspaper import Article
from ext_sum import summarize
from pytorch_pretrained_bert import BertConfig

# Uploadfile
import pdfplumber
import docx2txt

nltk.download('punkt')

st.set_page_config(page_title='Tóm tắt văn bản - Viện Công nghệ thông tin và Truyền thông - Đại học Bách Khoa Hà Nội',
    page_icon='soict-favicon.png')
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def remote_css(url):
    st.markdown(f'<link href="{url}" rel="stylesheet">', unsafe_allow_html=True)    

def icon(icon_name):
    st.markdown(f'<i class="material-icons">{icon_name}</i>', unsafe_allow_html=True)

def main():
    local_css('style.css')
    remote_css('https://fonts.googleapis.com/icon?family=Material+Icons')

    input_type = st.sidebar.selectbox(
        'Input Type:',
        ('URL', 'Raw Text', 'Upload File')
    )

    sum_level = st.sidebar.radio("Output Length: ", ["Short (3 sentences)", "Medium (5 sentences)"])
    max_length = 3 if sum_level == "Short (3 sentences)" else 5

    st.markdown("<h1 style='text-align: center;'>Fine-tune BERT for Extractive Summarization</h1>", unsafe_allow_html=True)

    # Load model
    model = load_model('bert_mlp')


    text = ""
    if input_type == "Raw Text":
        with open("./raw_data/input.txt") as f:
            sample_text = f.read()
        text = st.text_area("", sample_text, 400)
    elif input_type == "URL":
        url = st.text_input("", "https://edition.cnn.com/2021/05/12/business/ransomware-attacks-banks-stock-exchanges/index.html")
        st.markdown(f"<div class='original-news'><a href='{url}' target='_blank'>Read Original News</a></div>", unsafe_allow_html=True)
        text = crawl_url(url)
    else:
        docx_file = st.file_uploader("Upload File",type=['txt','docx','pdf'])
        if st.button("Process"):
            if docx_file is not None:
                if docx_file.type == "text/plain":
                    text = str(docx_file.read(),"utf-8")
                elif docx_file.type == "application/pdf":
                    try:
                        with pdfplumber.open(docx_file) as pdf:
                            page_numbers = len(pdf.pages)
                            for i in range(page_numbers):
                                page = pdf.pages[i]
                                text += " " + page.extract_text()
                    except:
                        st.write("None")
                elif docx_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                    text = docx2txt.process(docx_file)
                
                st.write(text)
    
    text_word_numbers = len(text.split())
    st.markdown(f"<div class='doc'>Document [{text_word_numbers} words]</div>", unsafe_allow_html=True)
    
    st.markdown(f"<hr>", unsafe_allow_html=True)
    with st.spinner('Wait for it...'):
        if text != "":
            input_fp = "./raw_data/input.txt"
            with open(input_fp, 'w') as file:
                file.write(text)

            result_fp = './results/summary.txt'
            summary = summarize(input_fp, result_fp, model, max_length=max_length)

            summary_word_numbers = len(summary.split())
            st.markdown(f"<h3 style='text-align: center;'>Summary [{summary_word_numbers} words]</h3>", unsafe_allow_html=True)
            st.markdown(f"<div align='justify' class='summary alert alert-success'>{summary}</div>", unsafe_allow_html=True)
        else:
            st.markdown("<h3 style='text-align: center;'>Summary</h3>", unsafe_allow_html=True)

@st.cache(suppress_st_warning=True)
def load_model(model_type):
    checkpoint = torch.load(f'checkpoints/{model_type}.pt', map_location='cpu')
    config = BertConfig.from_json_file('models/config.json')
    model = Summarizer(args=None, device="cpu", load_pretrained_bert=False, bert_config = config)
    model.load_cp(checkpoint)
    return model


def crawl_url(url):
    article = Article(url)
    article.download()
    try:
        article.parse()
    except :
        pass
    return article.text


if __name__ == "__main__":
    main()
