from io import BytesIO
from PyPDF2 import PdfReader # pdf 내용 추출
import pandas as pd
from openai.embeddings_utils import get_embedding, cosine_similarity # 텍스트 임베딩 API, 문장 간 유사성 계산
import openai
import os, re, tenacity, pickle, unicodedata
import pdfplumber
from konlpy.tag import Okt
from rank_bm25 import BM25Okapi
okt = Okt()

pdf_folder = '규정_외_문서/중요/업데이트' # PDF 파일이 있는 폴더 경로
pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith('.pdf')]
df_all = pd.DataFrame(columns=['text', 'CLS1', 'paper_title'])

def create_prompt(sum_text):
    system_role = f"""
    You are an AI language model.
    Your task is to organize the text in a document.
    Correct spelling, spacing, etc, and remove the document's table of contents, annexes, and appendices. 
    
    Here is the text in a document: """ + str(sum_text) + """
    
    You are required to respond in Korean. 
    """
    user_content = f"""Given the question: "{str(sum_text)}". """

    messages = [
        {"role": "system", "content": system_role},
        {"role": "user", "content": user_content}
    ]

    return messages

@tenacity.retry(wait=tenacity.wait_fixed(2), stop=tenacity.stop_after_attempt(3), reraise=True)
def gpt(messages):
    print('---')
    print('Sending request to GPT-3')
    r = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0,
        max_tokens=500,
    )
    answer=(
        r["choices"][0]
        .get("message")
        .get("content")
    )
    print('Done sending request to GPT-3')
    response = {'answer': answer}
    return response
    
for pdf_file in pdf_files:
    with pdfplumber.open('규정_외_문서/중요/업데이트/'+pdf_file) as pdf:
        sum_text = ''
        for page_number, page in enumerate(pdf.pages):
            print("------------------------------")
            print(f"Page {page_number + 1}:")

            # 페이지의 텍스트 블록을 순회합니다.
            for item in page.extract_words():
                text = list(item.values())[0]
                sum_text += (text + ' ')

# QA split
def split_into_chunks(text, max_chunk_size=500):
    """
    주어진 텍스트를 최대 길이가 max_chunk_size인 문장으로 분할한다.
    """
    chunks = []
    current_chunk = ""
    sentences = re.split("(?=Q\d+\))", text)

    for sentence in sentences:
        # 문장을 추가했을 때 chunk의 크기가 max_chunk_size를 초과하면, 현재까지의 chunk를 chunks에 추가
        if len(current_chunk) + len(sentence) > max_chunk_size:
            chunks.append(current_chunk)
            current_chunk = sentence
        else:
            current_chunk += sentence

    # 마지막에 남은 부분을 chunks에 추가
    chunks.append(current_chunk.strip())

    return chunks

chunk = split_into_chunks(sum_text, max_chunk_size=800)
df = pd.DataFrame(chunk)
add_text = '<<아래 내용은 대전 본원만 해당되며, 분원(정읍, 경주, 기장, 감포) 직원들에게는 해당되지 않습니다.>>\n'
df.columns = ['text']
df['CLS1'] = ['FnQ']*len(df)
df['paper_title'] = ['원자력안전관리실_대전']*len(df)
df['text'] = df.apply(lambda x: '[' + x['paper_title'] + '] ' + x['text'].strip(), axis=1)

## 임베딩 계산
embedding_model = "text-embedding-ada-002"
embeddings = df.iloc[:,0].apply([lambda x: get_embedding(x, engine=embedding_model)])
df["embeddings"] = embeddings
df.iloc[1:,0] = df.iloc[1:,:].apply(lambda x: add_text + x['text'].strip(), axis=1)

import pickle
with open('정제파일/embedding_Update_0523.pkl', 'rb') as f:
    df2 = pickle.load(f)
    
df_sum = pd.concat([df2, df])

df_sum.reset_index(inplace=True)
df_sum.drop(columns=['index'], inplace=True)

df2.iloc[2231,0] = df2.iloc[2231:].apply(lambda x: add_text + x['text'].strip(), axis=1)


# 저장하기
# with open('정제파일/embedding_Update_0523.pkl', 'wb') as f:
#     pickle.dump(df2, f)

def tokenizer(sent):
  sent = okt.morphs(sent, norm=False, stem=True)
  return sent

BM = [tokenizer(doc) for doc in df2.iloc[:,0]]

bm25 = BM25Okapi(BM)

# 저장
# with open('정제파일/bm25_Update_0523.pkl', 'wb') as f:
#     pickle.dump(bm25, f)
    

