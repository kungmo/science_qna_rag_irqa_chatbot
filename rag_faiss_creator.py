import os
import pandas as pd
import re
import codecs
import pickle
from kiwipiepy import Kiwi
from langchain_community.document_loaders import DataFrameLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_huggingface import HuggingFaceEmbeddings
from typing import List, Union
from langchain_core.documents import Document
import torch
import gc

# 가장 최근 파일 경로 찾는 함수
def get_newest_file_with_prefix(directory, prefix):
    # Get a list of all files in the directory
    files = os.listdir(directory)
    # Filter files that start with the specified prefix
    matching_files = [file for file in files if file.startswith(prefix)]
    if not matching_files:
        return None  # No matching files found
    # Get the full paths of the matching files
    matching_paths = [os.path.join(directory, file) for file in matching_files]
    # Sort the files by modification time (newest first)
    newest_file = max(matching_paths, key=os.path.getmtime)
    return newest_file

def drop_empty_rows_with_notna(df):
    # '질문' 열과 '답변' 열이 NaN이 아니고, 공백이 아닌 경우 필터링
    df_cleaned = df[df['질문'].notna() & df['답변'].notna() &
                    (df['질문'].str.strip() != '') & (df['답변'].str.strip() != '')]
    return df_cleaned.reset_index(drop=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

train_data_directory = os.path.join(BASE_DIR, "data")
qna_xlsx_location = get_newest_file_with_prefix(train_data_directory, 'df_qna_')
cus_xlsx_location = get_newest_file_with_prefix(train_data_directory, 'df_cus_')

# 고등학교 자료 df_h, 중학교 자료 df_m
df_qna = pd.read_excel(qna_xlsx_location, sheet_name='Sheet1')
df_cus = pd.read_excel(cus_xlsx_location, sheet_name='Sheet1')

df_qna = drop_empty_rows_with_notna(df_qna[['번호', '질문', '답변']])
df_cus = drop_empty_rows_with_notna(df_cus[['번호', '질문', '답변']])

df_qna = df_qna.drop_duplicates()
df_cus = df_cus.drop_duplicates()

def create_train_data(*dataframes):
    """
    여러 데이터프레임을 받아서 하나의 train_data를 생성하는 함수

    Parameters:
    *dataframes : 가변 개수의 pandas DataFrame
        처리할 데이터프레임들 ('질문'과 '답변' 컬럼이 있어야 함)

    Returns:
    pandas.DataFrame: 전처리된 train_data
    """
    # 입력된 데이터프레임이 없는 경우 처리
    if len(dataframes) == 0:
        raise ValueError("최소 하나 이상의 데이터프레임이 필요합니다.")

    try:
        # 데이터프레임 병합
        df_concat = pd.concat(dataframes, ignore_index=True)

        # 인덱스 초기화
        df_concat = df_concat.reset_index(drop=True)

        # 번호 컬럼 추가
        df_concat['번호'] = df_concat.index + 1

        # 필요한 컬럼만 선택
        df_concat = df_concat[['번호', '질문', '답변']]

        # 특수문자 ' 처리
        df_concat['답변'] = df_concat['답변'].str.replace("'", "")

        df_concat = df_concat.dropna()

        # 문서 컬럼 생성
        df_concat['문서'] = df_concat[['질문', '답변']].apply(lambda x: ' : '.join(map(str, x)), axis=1)

        return df_concat

    except Exception as e:
        raise Exception(f"데이터 처리 중 오류가 발생했습니다: {str(e)}")

train_data = create_train_data(df_qna, df_cus)

def create_docs_QNA(train_data, content_column="질문", chunk_size=1000, chunk_overlap=200):
    """
    train_data를 입력받아 docs_QNA를 생성하는 함수

    Parameters:
    train_data : pandas.DataFrame
        처리할 데이터프레임 ('질문'과 '답변' 컬럼이 필요)
    content_column : str, optional
        문서 생성에 사용할 컬럼 (default: "질문")
    chunk_size : int, optional
        텍스트 분할 크기 (default: 1000)
    chunk_overlap : int, optional
        텍스트 분할 중복 크기 (default: 200)

    Returns:
    list: 처리된 문서 리스트 (docs_QNA)
    """
    try:
        # 필요한 컬럼만 선택
        if not all(col in train_data.columns for col in ['질문', '답변']):
            raise ValueError("데이터프레임에 '질문'과 '답변' 컬럼이 모두 있어야 합니다.")

        processed_df = train_data[['질문', '답변']]

        # DataFrameLoader를 사용하여 문서 생성
        loader = DataFrameLoader(processed_df, page_content_column=content_column)
        documents_QNA = loader.load()

        # 텍스트 분할기 설정 및 문서 분할
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            encoding_name='cl100k_base'
        )

        # 문서 분할 실행
        docs_QNA = text_splitter.split_documents(documents_QNA)

        print(f"생성된 문서 수: {len(docs_QNA)}")
        return docs_QNA

    except Exception as e:
        raise Exception(f"문서 생성 중 오류가 발생했습니다: {str(e)}")

docs_QNA = create_docs_QNA(train_data)


def save_pickle(obj, file_path: str) -> None:
    """
    객체를 pickle 파일로 저장하는 함수

    Parameters:
        obj: 저장할 객체
        file_path (str): 저장할 파일 경로 (.pkl 확장자 포함)
    """
    # 디렉토리가 없으면 생성
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # 객체를 pickle 파일로 저장
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f)
    print(f"저장 완료: {file_path}")

save_pickle(docs_QNA, os.path.join(BASE_DIR, "docs_QNA", "docs_QNA.pkl"))

from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_huggingface import HuggingFaceEmbeddings
from typing import List, Union
from langchain_core.documents import Document
import os
import torch
import gc


def clear_gpu_memory():
    """
    GPU 메모리를 철저하게 정리하는 함수
    """
    # 1. 모든 PyTorch 캐시를 비움
    torch.cuda.empty_cache()

    # 2. 가비지 컬렉터 실행
    gc.collect()

    # 3. 현재 스코프의 모든 변수를 확인하고 GPU 텐서 제거
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj):
                del obj
            elif hasattr(obj, 'data') and torch.is_tensor(obj.data):
                del obj.data
        except:
            pass

    # 4. CUDA 캐시를 다시 한번 비움
    torch.cuda.empty_cache()

    # 5. 가능한 경우 GPU 리셋
    try:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_accumulated_memory_stats()
    except:
        pass


def cleanup_model(embeddings_model):
    """
    임베딩 모델의 리소스를 정리하는 함수
    """
    try:
        if hasattr(embeddings_model, 'client'):
            # 모델을 CPU로 이동
            if hasattr(embeddings_model.client, 'model'):
                embeddings_model.client.model.cpu()

            # 모델 객체 참조 제거
            if hasattr(embeddings_model.client, 'model'):
                embeddings_model.client.model = None

            # client 객체 참조 제거
            embeddings_model.client = None
    except Exception as e:
        print(f"Warning during model cleanup: {e}")
    finally:
        # embeddings_model 참조 제거
        embeddings_model = None


def create_and_save_vectorstore(
        documents: List[Document],
        save_path: str,
        model_name: str = 'nlpai-lab/KURE-v1',
        device: str = 'cpu', # 'cuda' 또는 'cpu'
        normalize_embeddings: bool = True
) -> FAISS:
    """
    문서를 벡터스토어로 변환하고 지정된 경로에 저장하는 함수
    GPU 메모리를 완전히 정리합니다.

    Args:
        documents: 변환할 문서 리스트
        save_path: 벡터스토어를 저장할 경로
        model_name: 임베딩 모델 이름
        device: 사용할 디바이스 ('cuda' 또는 'cpu')
        normalize_embeddings: 임베딩 정규화 여부

    Returns:
        생성된 FAISS 벡터스토어 객체
    """
    embeddings_model = None
    vectorstore = None

    try:
        # 임베딩 모델 초기화
        embeddings_model = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': device},
            encode_kwargs={'normalize_embeddings': normalize_embeddings}
        )

        # 벡터스토어 생성
        vectorstore = FAISS.from_documents(
            documents,
            embedding=embeddings_model
        )

        # 저장 디렉토리가 없으면 생성
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # 벡터스토어 저장
        vectorstore.save_local(save_path)

        return vectorstore

    finally:
        # GPU 메모리 정리
        if device == 'cuda':
            if embeddings_model is not None:
                cleanup_model(embeddings_model)
            clear_gpu_memory()


def save_multiple_vectorstores(base_path: str, docs_dict: dict) -> dict:
    """
    여러 문서 집합을 각각의 벡터스토어로 변환하고 저장하는 함수

    Args:
        base_path: 기본 저장 경로
        docs_dict: {'이름': documents} 형태의 딕셔너리

    Returns:
        생성된 벡터스토어들의 딕셔너리
    """
    vectorstores = {}
    try:
        for name, docs in docs_dict.items():
            save_path = os.path.join(base_path, f'faiss_{name}')
            vectorstores[name] = create_and_save_vectorstore(docs, save_path)
        return vectorstores
    finally:
        # 추가적인 메모리 정리
        if torch.cuda.is_available():
            clear_gpu_memory()


# 벡터스토어 생성 후 메모리 정리를 위한 함수
def cleanup_after_vectorstore():
    """
    벡터스토어 생성 후 호출하여 메모리를 정리하는 함수
    """
    clear_gpu_memory()

# FAISS DB 저장
vectorstore_QNA = create_and_save_vectorstore(documents=docs_QNA, save_path=os.path.join(BASE_DIR, "faiss_QNA"))