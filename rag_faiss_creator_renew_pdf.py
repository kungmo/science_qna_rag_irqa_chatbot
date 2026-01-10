import os
import pandas as pd
import re
import codecs
import pickle
from kiwipiepy import Kiwi
import sys
from pypdf import PdfReader
from langchain_core.documents import Document

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def get_pdf_file_paths(directory):
    # Get the absolute path of the directory
    abs_directory = os.path.abspath(directory)

    # Get a list of all items in the directory
    all_items = os.listdir(abs_directory)

    # Filter for files (not directories) with .pdf extension and get their full paths
    pdf_files = [os.path.join(abs_directory, file) for file in all_items
                 if os.path.isfile(os.path.join(abs_directory, file)) and file.lower().endswith('.pdf')]

    return pdf_files

directory = os.path.join(BASE_DIR, "pdfs")
pdf_paths = get_pdf_file_paths(directory)


def clean_text(text):
    # 1. 특수 문자 제거 (≐ 등)
    text = re.sub(r'≐', '', text)

    # 2. 탭과 다중 공백을 단일 공백으로 변환
    text = re.sub(r'[ \t]+', ' ', text)

    # 3. 불필요한 줄 바꿈 제거
    # 예: 단어가 끊어져 이어지는 경우를 고려하여 처리
    # 여기서는 간단히 모든 줄 바꿈을 공백으로 대체
    text = re.sub(r'\n+', '\n', text)  # 여러 줄 바꿈을 단일 줄 바꿈으로
    text = re.sub(r'\s*\n\s*', '\n', text)  # 줄 앞뒤의 공백 제거

    # 4. 불필요한 숫자나 코드 제거 (예: '각 열기11174~175')
    # 필요에 따라 수정 가능
    text = re.sub(r'\d+~\d+', '', text)
    text = re.sub(r'\d+', '', text)

    # 5. 불필요한 쉼표 및 공백 정리
    text = re.sub(r',+', ',', text)
    text = re.sub(r'\s+,', ',', text)
    text = re.sub(r',\s+', ',', text)
    text = re.sub(r'\s+', ' ', text)

    # 6. 앞뒤 공백 제거
    text = text.strip()

    return text


def decode_unicode_escapes(text):
    # 정규표현식을 사용하여 유니코드 이스케이프 문자열과 \x 문자열을 찾아 변환합니다.
    def unicode_replace(match):
        return codecs.decode(match.group(0), 'unicode_escape')

    # 공백 처리를 위해 \u200c를 공백으로 변환합니다.
    decoded_text = re.sub('\u200c', ' ', text)
    # \uXXXX 패턴을 해석합니다.
    decoded_text = re.sub(r'\\u([a-fA-F0-9]{4})', unicode_replace, decoded_text)
    # \xXX 패턴을 해석합니다.
    decoded_text = re.sub(r'\\x([a-fA-F0-9]{2})', lambda x: chr(int(x.group(1), 16)), decoded_text)
    # \n을 공백으로 처리 후, 뒤에 붙는 문자열 제거
    decoded_text = re.sub(r'\\n', ' ', decoded_text)
    # 연속된 공백을 하나의 공백으로 대체
    decoded_text = re.sub(r'\s+', ' ', decoded_text)
    # 6자리 이상의 숫자를 제거
    decoded_text = re.sub(r'\d{6,}', ' ', decoded_text)
    return decoded_text


# 선다형 문제 패턴을 인식하는 함수 추가
def is_multiple_choice_question(text):
    # 1. 선다형 문제의 전형적인 패턴(원문자와 ㄱ,ㄴ,ㄷ의 조합)을 찾기
    # 예: ① ㄴ ② ㄷ ③ ㄱ, ㄴ ④ ㄱ, ㄷ ⑤ ㄱ, ㄴ, ㄷ
    mcq_full_pattern = r'[①②③④⑤]\s*[ㄱㄴㄷㄹㅁ,\s]+[①②③④⑤]\s*[ㄱㄴㄷㄹㅁ,\s]+'

    # 2. 3개 이상의 원문자가 한 문장에 있는 경우 (선다형 문제일 확률이 높음)
    mcq_circles_pattern = r'[①②③④⑤][^①②③④⑤]{0,10}[①②③④⑤][^①②③④⑤]{0,10}[①②③④⑤]'

    # 3. 선다형 문제에 나오는 전형적인 표현 확인
    mcq_keywords = ['다음 중', '옳은 것은', '옳지 않은 것은', '고르시오', '모두 고른 것은']
    has_mcq_keywords = any(keyword in text for keyword in mcq_keywords)

    # 4. ㄱ, ㄴ, ㄷ이 일정 패턴으로 나타나고 원문자가 있는 경우
    options_pattern = r'ㄱ[,\s.]+ㄴ[,\s.]+ㄷ'
    has_circles = any(c in text for c in '①②③④⑤')

    # 점수 기반 시스템 구현
    score = 0

    # 원문자(①,②,③ 등) 개수 확인
    circle_count = len(re.findall(r'[①②③④⑤]', text))
    if circle_count >= 3:
        score += 3
    elif circle_count == 2:
        score += 1

    # ㄱ,ㄴ,ㄷ 패턴 확인 (ㄱ, ㄴ, ㄷ이 모두 있는 경우)
    if 'ㄱ' in text and 'ㄴ' in text and 'ㄷ' in text:
        score += 2

    # 선다형 문제 키워드 확인
    for keyword in mcq_keywords:
        if keyword in text:
            score += 2
            break

    # 전형적인 선다형 문제 패턴 확인
    if re.search(mcq_full_pattern, text):
        score += 4

    if re.search(mcq_circles_pattern, text):
        score += 3

    if re.search(options_pattern, text) and has_circles:
        score += 3

    # 선다형 문제 판단 결과 반환
    return (score >= 4 or
            re.search(mcq_full_pattern, text) or
            re.search(mcq_circles_pattern, text) or
            (has_mcq_keywords and has_circles) or
            (re.search(options_pattern, text) and has_circles))


# 이상하게 띄어 쓰는 용어는 사용자 사전 추가
kiwi = Kiwi(num_workers=16)
kiwi.add_user_word('변화량', 'NNP')
kiwi.add_user_word('운동량', 'NNP')
kiwi.add_user_word('충격량', 'NNP')

# 제외할 문자열 리스트. 교과서에서 과학 내용이 아닌 활동과 관련된 문구가 들어 있는 문장을 삭제하기 위해서 사용.
exclude_strings = ["\x01", "\x07", "을까?", "해 보자", "해 보기", "지도한다", "생각하게 한다", "스스로 확인하기", "할 수 있다.",
                   "진도 체크", "하였나요?", "했나요?", "활동 후기", "스스로 점검하기", "마무리", "써 보자", "도움 영상",
                   "고른 것은?", "쓰시오", "모두 고른", "활동 도움", "유의 사항", "유의할 점", "준비물", "나타낸 것이다.",
                   "실시한다.", "보고서 작성하기", "발표하기", "( )", "ㄱ.", "ㄴ.", "ㄷ.", "않도록 한다", "학생들이",
                   "나타내 보자", "주의한다", "고르시오", "모두 적는다.", "탐구 능력", "지도상의", "유의점", "알아보자",
                   "만들어 보자", "통합 자료실", "지도 방안", "스스로 평가하기", "수업 지도안", "수행하였는가?", "제시하였는가?",
                   "만들었는가?", "탐구를 진행한다.", "탐구 기능", ".indd", ".indb", "토의하는", "다르기 때문인가?"]

# ㄱ. 등이 두 개 이상 포함된 경우 내용 삭제
patterns = ["ㄱ.", "(가)", "ㄴ.", "(나)", "ㄷ.", "(다)", "거짓", "아니요"]

PDF_sentences = []
sentence_buffer = []

for pdf_path in pdf_paths:
    documents_PDF = PdfReader(pdf_path)
    reference = pdf_path.replace(directory + "/", "")
    for i in range(len(documents_PDF.pages) - 1, -1, -1):
        text = documents_PDF.pages[i].extract_text(0)
        text = decode_unicode_escapes(text)
        cleaned_text = clean_text(text)
        split = kiwi.split_into_sents(cleaned_text)
        sentence_buffer = []

        for sentence in split:
            # 제외 문자열 체크 및 문장 길이 체크
            if any(exclude_string in sentence.text for exclude_string in exclude_strings):
                continue

            # 새로 추가: 선다형 문제 패턴 체크
            if is_multiple_choice_question(sentence.text):
                continue

            # 기존 코드: ㄱ. 등이 두 개 이상 포함된 경우 내용 삭제
            if any(sentence.text.count(pattern) > 2 for pattern in patterns):
                continue

            # 문제 관련 특정 문구 체크
            problem_phrases = ["옳지 않은 것은?", "옳은 것은?", "고른 것은?",
                               "옳지 않은 것을 고른 것은?", "보기 중", "보기에서"]

            if any(phrase in sentence.text for phrase in problem_phrases):
                # 선다형 문제를 포함하는 문장이 발견되면 현재 버퍼를 비움
                # 이렇게 하면 이 문장을 포함한 모든 문장 모음이 폐기됨
                sentence_buffer = []
                continue

            # 문장이 짧으면 교과서에서 뭔가 묻는 경우가 많으므로 공백 포함 15글자 이상의 문장만 수집.
            if len(sentence.text) >= 15:
                sentence_buffer.append(sentence.text)

            # 버퍼 길이 체크
            if len(sentence_buffer) == 6:
                combined_text = " ".join(sentence_buffer)
                combined_text = kiwi.space(combined_text, reset_whitespace=True)
                PDF_sentences.append(Document(page_content=combined_text, metadata={"source": reference}))
                sentence_buffer = sentence_buffer[2:]  # 앞의 2개 문장을 buffer에서 제거하고, 뒤의 4개 문장만 남김

        # 문서의 마지막 부분 처리
        if sentence_buffer:
            combined_text = " ".join(sentence_buffer)
            combined_text = kiwi.space(combined_text, reset_whitespace=True)
            PDF_sentences.append(Document(page_content=combined_text, metadata={"source": reference}))

docs_PDF = PDF_sentences

output_filename = os.path.join(BASE_DIR, "docs_PDF", "docs_PDF.pkl")

with open(output_filename, 'wb') as f:
    pickle.dump(docs_PDF, f)

print(f'Saved {len(docs_PDF)} document chunks to {output_filename}')

# 국립국어원 사전에서 과학 관련 뜻풀이만 추리기

Korean_dictionary_directory = os.path.join(BASE_DIR, "Korean_dictionary")

def get_file_names(directory):
    # Get a list of all files in the directory
    files = os.listdir(directory)
    file_locations = []
    df_all = pd.DataFrame()
    for file in files:
        file_locations.append(os.path.join(directory, file))
        df = pd.read_excel(os.path.join(directory, file))
        df = df[['어휘', '뜻풀이', '전문 분야']]
        df = df.dropna(axis=0)
        df_all = pd.concat([df_all, df], ignore_index = True)
    return df_all

df_dic = get_file_names(Korean_dictionary_directory)

science_category_list = ['물리', '화학', '생명', '지구', '수학', '해양', '건설', '교육', '공업', '광업', '정보',
                        '통신', '교통', '전기', '전자', '기계', '약학', '의학', '보건', '식물', '의학', '임업',
                        '지리', '천문', '환경']
science_category = '|'.join(science_category_list)

df_dic = df_dic[df_dic['전문 분야'].str.contains(science_category)]

df_dic = df_dic[['어휘', '뜻풀이']]

# 정규표현식 패턴
pattern_space = r"\^"
pattern_blank = r"-|\(\d+\)"

# 특정 패턴을 대체하여 제거
df_dic['어휘'] = df_dic['어휘'].str.replace(pattern_space, ' ', regex=True)
df_dic['어휘'] = df_dic['어휘'].str.replace(pattern_blank, '', regex=True)

df_dic['뜻풀이'] = df_dic['뜻풀이'].str.replace('\n', " ")

df_dic['문서'] = df_dic[['어휘', '뜻풀이']].apply(lambda x: '의 사전적 의미 : '.join(map(str, x)), axis=1)

# Indexing (Texts -> Embedding -> Store)
from langchain_community.document_loaders import DataFrameLoader

loader = DataFrameLoader(df_dic, page_content_column="문서")

docs_dic = loader.load()

# 벡터스토어에 문서 임베딩을 저장
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_huggingface import HuggingFaceEmbeddings

embeddings_model = HuggingFaceEmbeddings(
    model_name='nlpai-lab/KURE-v1',
    model_kwargs={'device':'cpu'}, # 'cuda' 또는 'cpu'
    encode_kwargs={'normalize_embeddings':True},
)

vectorstore_PDF = FAISS.from_documents(docs_PDF, embedding = embeddings_model)
vectorstore_dic = FAISS.from_documents(docs_dic, embedding = embeddings_model)

# FAISS DB 저장
vectorstore_PDF.save_local(os.path.join(BASE_DIR, "faiss_PDF"))
vectorstore_dic.save_local(os.path.join(BASE_DIR, "faiss_dic"))