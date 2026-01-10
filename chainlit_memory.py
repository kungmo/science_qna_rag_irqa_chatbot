import chainlit as cl
from chainlit.config import config
from langchain_community.vectorstores import FAISS
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage, trim_messages
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv
from pathlib import Path
import time
import datetime
import base64
import asyncio
from typing import Dict, Any, List, Optional
import pymysql.cursors
import tempfile
import shutil
from PIL import Image, UnidentifiedImageError
import io
import magic
import pillow_heif
import json
import opendataloader_pdf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 환경 변수 설정
load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
IMAGES_DIR = os.path.join(BASE_DIR, "uploaded_images")
PDFS_DIR = os.path.join(BASE_DIR, "uploaded_pdfs")
PDF_TEMP_DIR = os.path.join(BASE_DIR, "pdf_temp")

os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(PDFS_DIR, exist_ok=True)
os.makedirs(PDF_TEMP_DIR, exist_ok=True)

# 여기서 LLM을 선택할 수 있습니다. 모두 바꾸기로 llm_gemini와 llm_local_server를 전환하세요.
llm_gemini = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", temperature=1.0, thinking_level="high")
llm_local_server = ChatOllama(model="gemma3:12b-it-qat", temperature=0)
#llm_gemma_API = ChatGoogleGenerativeAI(model="gemma-3-27b", temperature=0)

def format_qna_docs(docs: List[Document]) -> List[str]:
    """QNA 문서를 포매팅"""
    formatted = []
    for doc in docs:
        if doc.metadata.get("답변"):
            qna = f"질문: {doc.page_content}\n답변: {doc.metadata['답변']}"
            formatted.append(qna)
    return formatted


def format_pdf_docs(docs: List[Document]) -> List[str]:
    """PDF 문서를 포매팅"""
    formatted = []
    for doc in docs:
        content = f"출처: {doc.metadata.get('source', '교과서')}\n내용: {doc.page_content}"
        formatted.append(content)
    return formatted


def create_context_messages(qna_docs: List[Document] = [], pdf_docs: List[Document] = [],
                            uploaded_pdf_docs: List[Document] = None) -> List[BaseMessage]:
    """QNA와 PDF 문서를 별도의 시스템 메시지로 변환"""
    context_messages = []

    # 방금 업로드된 PDF 내용이 있으면 추가 (우선순위 1순위)
    if uploaded_pdf_docs and len(uploaded_pdf_docs) > 0:
        uploaded_pdf_context = "학생이 업로드한 PDF로부터 추출한 텍스트:\n\n"
        for doc in uploaded_pdf_docs:
            chunk_text = f"출처: {doc.metadata.get('source', '학생이 업로드한 PDF')}\n내용: {doc.page_content}"
            uploaded_pdf_context += chunk_text + "\n\n"
        context_messages.append(SystemMessage(content=uploaded_pdf_context))

    if qna_docs and len(qna_docs) > 0:
        qna_formatted = format_qna_docs(qna_docs)
        qna_context = "관련된 과거 질문-답변:\n\n" + "\n\n".join(qna_formatted)
        context_messages.append(SystemMessage(content=qna_context))

    if pdf_docs and len(pdf_docs) > 0:
        pdf_formatted = format_pdf_docs(pdf_docs)
        pdf_context = "교과서 관련 내용:\n\n" + "\n\n".join(pdf_formatted)
        context_messages.append(SystemMessage(content=pdf_context))

    return context_messages


# 이미지 변환 함수
def convert_image_to_jpeg(image_path):
    """
    다양한 이미지 형식을 JPEG로 변환 (HEIC 지원 강화 및 16bit 이미지 대응)
    """
    try:
        # 파일 타입 감지
        mime = magic.Magic(mime=True)
        file_type = mime.from_file(image_path)
        file_type_lower = file_type.lower()

        # [디버그] 파일 정보 터미널 출력
        print(f"[Debug] Converting image: {image_path} (MIME: {file_type})")

        img = None

        # HEIC/HEIF 파일인 경우 (확장자 또는 MIME 타입으로 확인)
        if ("heic" in file_type_lower or "heif" in file_type_lower) or image_path.lower().endswith(('.heic', '.heif')):
            # pillow_heif를 사용하여 명시적으로 로드 (HDR -> 8bit 변환)
            heif_file = pillow_heif.open_heif(image_path, convert_hdr_to_8bit=True)
            img = heif_file.to_pillow()
        else:
            # 일반 이미지는 바로 열기
            img = Image.open(image_path)

        # [핵심 수정] 이미지 모드 변환 (I;16, LAB, P, RGBA 등을 RGB로 강제 변환)
        # JPEG는 16비트나 투명도(Alpha)를 지원하지 않으므로 변환이 필수입니다.
        if img.mode in ('RGBA', 'P', 'LA', 'I', 'I;16', 'L'):
            img = img.convert('RGB')

        # 메모리 버퍼에 JPEG로 저장
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        buffer.seek(0)

        # base64 인코딩
        jpeg_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

        return jpeg_base64, None
    except Exception as e:
        import traceback
        traceback_str = traceback.format_exc()
        # [디버그] 오류 발생 시 터미널에 빨간색(혹은 눈에 띄게) 출력
        print(f"\n❌ [ERROR] convert_image_to_jpeg failed:\n{traceback_str}\n")
        return None, f"이미지 변환 중 오류 발생: {str(e)}\n{traceback_str}"


def save_image_file(image_path, user_name):
    try:
        # 파일 타입 감지
        mime = magic.Magic(mime=True)
        file_type = mime.from_file(image_path)
        file_type_lower = file_type.lower()

        # 타임스탬프 생성
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        sanitized_username = user_name.replace(" ", "_").replace("/", "_").replace("\\", "_")

        # HEIC/HEIF 파일인 경우 JPEG로 변환하여 저장
        if ("heic" in file_type_lower or "heif" in file_type_lower) or image_path.lower().endswith(('.heic', '.heif')):
            # 직접 로드 및 8bit 변환
            heif_file = pillow_heif.open_heif(image_path, convert_hdr_to_8bit=True)
            img = heif_file.to_pillow()

            # JPEG로 저장 파일명 설정
            new_filename = f"{timestamp}_{sanitized_username}.jpg"
            save_path = os.path.join(IMAGES_DIR, new_filename)

            # [핵심 수정] 16비트 등을 8비트 RGB로 변환
            if img.mode in ('RGBA', 'P', 'LA', 'I', 'I;16', 'L'):
                img = img.convert('RGB')

            img.save(save_path, format="JPEG")
        else:
            # 기타 이미지 파일은 원래 확장자 유지
            original_extension = Path(image_path).suffix
            if not original_extension:
                if "jpeg" in file_type_lower or "jpg" in file_type_lower:
                    original_extension = ".jpg"
                elif "png" in file_type_lower:
                    original_extension = ".png"
                else:
                    original_extension = ".jpg"

            extension = original_extension
            new_filename = f"{timestamp}_{sanitized_username}{extension}"
            save_path = os.path.join(IMAGES_DIR, new_filename)
            shutil.copy2(image_path, save_path)

        return save_path, None
    except Exception as e:
        import traceback
        traceback_str = traceback.format_exc()
        # [디버그] 터미널 출력
        print(f"\n❌ [ERROR] save_image_file failed:\n{traceback_str}\n")
        return None, f"이미지 저장 중 오류 발생: {str(e)}\n{traceback_str}"


def save_pdf_file(pdf_path, user_name):
    try:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        sanitized_username = user_name.replace(" ", "_").replace("/", "_").replace("\\", "_")
        new_filename = f"{timestamp}_{sanitized_username}.pdf"
        save_path = os.path.join(PDFS_DIR, new_filename)
        shutil.copy2(pdf_path, save_path)
        return save_path, None
    except Exception as e:
        import traceback
        traceback_str = traceback.format_exc()
        return None, f"PDF 저장 중 오류 발생: {str(e)}\n{traceback_str}"


def extract_text_from_pdf_with_opendataloader(pdf_path, pdf_filename):
    """opendataloader_pdf를 사용하여 한글 PDF 텍스트 추출

    사용자가 업로드한 PDF는 간단한 처리로 진행 (문단 길이 필터링 없음)
    """
    try:
        temp_output_dir = os.path.join(PDF_TEMP_DIR, f"temp_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(temp_output_dir, exist_ok=True)

        # PDF를 JSON으로 변환
        opendataloader_pdf.convert(
            input_path=[pdf_path],
            output_dir=temp_output_dir,
            format=["json"]
        )

        # 변환된 JSON 파일 찾기
        json_files = [f for f in os.listdir(temp_output_dir) if f.endswith('.json')]
        if not json_files:
            return None, "JSON 파일이 생성되지 않았습니다."

        json_file_path = os.path.join(temp_output_dir, json_files[0])

        # JSON 파일 읽기
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 텍스트 추출 (간단한 처리)
        documents = []

        if 'kids' not in data:
            return None, "PDF 파일에 내용이 들어 있지 않습니다."

        current_heading = ""
        current_content_parts = []

        for kid in data['kids']:
            if kid['type'] == 'heading':
                # 이전 내용이 있으면 저장
                if current_content_parts:
                    full_content = current_heading + "\n" + " ".join(current_content_parts)
                    if len(full_content.replace(" ", "")) >= 10:
                        doc = Document(
                            page_content=full_content,
                            metadata={"source": pdf_filename}
                        )
                        documents.append(doc)

                current_heading = kid['content']
                current_content_parts = []

            elif kid['type'] == 'paragraph':
                current_content_parts.append(kid['content'])

        # 마지막 내용 저장
        if current_content_parts:
            full_content = current_heading + "\n" + " ".join(current_content_parts)
            if len(full_content.replace(" ", "")) >= 10:
                doc = Document(
                    page_content=full_content,
                    metadata={"source": pdf_filename}
                )
                documents.append(doc)

        # 추출된 청크가 길면 추가로 분할
        if documents:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=100,
                length_function=len,
                separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
            )

            split_documents = []
            for doc in documents:
                chunks = text_splitter.split_text(doc.page_content)
                for chunk in chunks:
                    split_documents.append(
                        Document(
                            page_content=chunk,
                            metadata={"source": pdf_filename}
                        )
                    )
            documents = split_documents

        # 임시 디렉토리 정리
        shutil.rmtree(temp_output_dir, ignore_errors=True)

        if not documents:
            return None, "PDF에서 추출된 텍스트가 없습니다."

        return documents, None

    except Exception as e:
        import traceback
        traceback_str = traceback.format_exc()
        # 임시 디렉토리 정리
        if 'temp_output_dir' in locals():
            shutil.rmtree(temp_output_dir, ignore_errors=True)
        return None, f"PDF 텍스트 추출 중 오류 발생: {str(e)}\n{traceback_str}"


def create_dynamic_system_prompt(
        user_name: str = "미등록",
        use_pdf_only: bool = False,
        uploaded_pdf_docs: Optional[List[Document]] = None
) -> SystemMessage:
    """
    사용자 이름과 PDF 여부에 따라 동적으로 시스템 프롬프트 생성
    """

    if not user_name or user_name == "미등록":
        name_context = ""
    else:
        name_context = f"'{user_name}'님의 "

    # PDF가 있는 경우
    if use_pdf_only and uploaded_pdf_docs:
        return SystemMessage(content=f"""<task>{name_context}질문에 대해 200단어 이내로 정확하고 명확한 답변을 제공</task>
<format_instructions>
    1. Markdown 형식으로 출력
    2. 수식 작성 시 LaTeX 표기법을 엄격히 준수
        <good_example>
            (1) $$ \\alpha = \\frac{{\\text{{분자}}}}{{\\text{{분모}}}} \\times 100% $$ 
            (2) $$ \\text{{Ag}}^+ $$
        </good_example>
    3. URL은 plain text로 표현
</format_instructions>
<output_instructions>
    1. '생각'과 '답변' 두 부분으로 나누어 응답
    2. 학생이 업로드한 PDF에서 추출한 텍스트 내용을 근거로 답변
    3. PDF 내용이 질문에 직접 관련이 없는 경우, "죄송합니다. 첨부하신 PDF 파일에는 이 질문에 대한 답변이 포함되어 있지 않습니다."라고 답변
    4. 그림 파일과 PDF 파일이 모두 첨부된 경우에는 PDF 내용을 우선적으로 참조하여 답변
    5. 질문과 관련이 있는 과거 질문-답변 내용이 있다면 질문-답변 내용을 인용하면서 근거를 들어 답변
    6. 질문과 관련이 있는 교과서 내용이 있다면 교과서 내용을 인용하면서 근거로 들어 답변
    7. 고등학교 1학년 학생이 이해할 수 있는 적절한 용어 사용
</output_instructions>""")

    # PDF가 없는 경우 (일반 과학 질문)
    else:
        return SystemMessage(content=f"""<task>{name_context}과학 관련 질문에 대한 200단어 이내로 정확하고 명확한 답변 제공</task>
<format_instructions>
    1. Markdown 형식으로 출력
    2. 수식 작성 시 LaTeX 표기법을 엄격히 준수
        <good_example>
            (1) $$ \\alpha = \\frac{{\\text{{분자}}}}{{\\text{{분모}}}} \\times 100% $$ 
            (2) $$ \\text{{Ag}}^+ $$
        </good_example>
    3. URL은 plain text로 표현
</format_instructions>
<output_instructions>
    1. '생각'과 '답변' 두 부분으로 나누어 응답
    2. 질문과 관련이 있는 과거 질문-답변 내용이 있다면 가장 우선적으로 참조하여 인용하면서 근거를 들어 답변
    3. 질문과 관련이 있는 교과서 내용이 있다면 교과서 내용을 인용하면서 근거로 들어 답변
    4. 관련 정보가 없다면 LLM의 자체 지식을 사용하여 답변
    5. 그림 파일을 첨부했다면 그림 파일의 내용을 분석하고 그 내용을 바탕으로 답변
    6. 학생이 업로드한 PDF로부터 추출한 텍스트가 있다면 그 내용을 바탕으로 질문에 답변
    7. 답변할 수 없는 경우 "죄송합니다. 제가 알지 못하는 내용입니다."라고 답변
    8. 고등학교 1학년 학생이 이해할 수 있는 적절한 용어 사용
</output_instructions>""")

# 임베딩 모델 초기화
embeddings_model = HuggingFaceEmbeddings(
    model_name='nlpai-lab/KURE-v1',
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True},
)

# Vector stores 로드
vectorstore_QNA = FAISS.load_local(os.path.join(BASE_DIR, "faiss_QNA"),
                                   embeddings_model,
                                   allow_dangerous_deserialization=True,
                                   )
vectorstore_QNA.embedding_function = embeddings_model.embed_query

vectorstore_PDF = FAISS.load_local(os.path.join(BASE_DIR, "faiss_PDF"),
                                   embeddings_model,
                                   allow_dangerous_deserialization=True,
                                   )
vectorstore_PDF.embedding_function = embeddings_model.embed_query


@cl.on_chat_start
async def start():
    welcome_messages = [
        ["안녕하세요?", "저는", "선배들이", "했던", "질문과", "과학", "교재", "내용을", "바탕으로", "여러분의", "질문에", "답변하는", "생성형", "AI",
         "챗봇입니다.\n\n",
         "과학공부", "하시면서", "궁금한", "것을", "물어보세요.", "중학교와", "고등학교", "과학", "질문과", "학교", "생활과", "관련된", "질문에", "답해줄", "수",
         "있습니다.\n\n",
         "저는", "그림이나", "사진을", "인식할", "수", "있으며,", "PDF 파일을", "업로드하면", "그", "내용도", "분석할", "수", "있습니다.",
         "최근", "대화를", "30개까지", "기억합니다.", "추가 질문이", "있다면", "질문해 주세요.🎬\n\n",
         "AI", "언어", "모델의", "특성상", "제가", "쓴", "답변이", "정확하지", "않을", "수", "있으므로", "교과서와", "참고서에서", "관련된", "내용을", "찾아보시기를",
         "권합니다.😊"]
    ]

    for msg in welcome_messages:
        collected_msg = cl.Message(content="", author="science_chatbot")
        for token in msg:
            time.sleep(0.015)
            await collected_msg.stream_token(token + " ")
        await collected_msg.send()

    res = await cl.AskActionMessage(
        content="별명을 알려주세요.",
        actions=[
            cl.Action(name="continue", payload={"value": "continue"}, label="✅ 알려주기"),
            cl.Action(name="cancel", payload={"value": "cancel"}, label="❌ 건너뛰기"),
        ],
        author="science_chatbot"
    ).send()

    if res and res.get("payload").get("value") == "continue":
        res_2 = await cl.AskUserMessage(content="별명을 알려주세요.", timeout=30, author="science_chatbot").send()
        if res_2:
            user_name = res_2['output'].replace("'", "")
            cl.user_session.set("user_name", user_name)
            await cl.Message(content=f"안녕하세요, {user_name}님! 무엇이 궁금하신가요?", author="science_chatbot").send()
    if res and res.get("payload").get("value") == "cancel":
        cl.user_session.set("user_name", "미등록")
        await cl.Message(content="안녕하세요! 무엇이 궁금하신가요?", author="science_chatbot").send()

    retriever_QNA = vectorstore_QNA.as_retriever(
        search_type='mmr',
        search_kwargs={'k': 4, 'fetch_k': 40, 'lambda_mult': 0.95}
    )

    retriever_PDF = vectorstore_PDF.as_retriever(
        search_type='mmr',
        search_kwargs={'k': 6, 'fetch_k': 60, 'lambda_mult': 0.95}
    )

    history_dict = {}

    def get_session_history(session_id: str):
        if session_id not in history_dict:
            history_dict[session_id] = InMemoryChatMessageHistory()
        return history_dict[session_id]

    cl.user_session.set("session_id", cl.user_session.get("id"))
    cl.user_session.set("get_session_history", get_session_history)
    cl.user_session.set("retriever_QNA", retriever_QNA)
    cl.user_session.set("retriever_PDF", retriever_PDF)
    cl.user_session.set("messages", [])
    cl.user_session.set("uploaded_pdf_docs", [])


@cl.on_message
async def main(message: cl.Message):
    messages = cl.user_session.get("messages")
    uploaded_pdf_docs = cl.user_session.get("uploaded_pdf_docs", [])
    user_name = cl.user_session.get("user_name", "미등록")

    def load_memory(input_dict: Dict[str, Any]) -> List[BaseMessage]:
        return trim_messages(messages, token_counter=len, max_tokens=60,
                             strategy="last", start_on="human", include_system=True, allow_partial=False)

    images = []
    pdfs = []

    for file in message.elements:
        if hasattr(file, 'mime') and hasattr(file, 'path') and hasattr(file, 'name'):
            file_path_lower = file.path.lower()
            file_name_lower = file.name.lower()

            # 이미지 파일 판단: MIME 타입 또는 원본 파일명 확장자 기반
            is_image = (
                    "image" in file.mime or
                    file_name_lower.endswith(('.jpg', '.jpeg', '.png', '.gif', '.heic', '.heif', '.bmp', '.webp'))
            )

            # PDF 파일 판단
            is_pdf = "pdf" in file.mime or file_name_lower.endswith('.pdf')

            if is_image and not is_pdf:
                images.append(file)
            elif is_pdf:
                pdfs.append(file)

    image_base64 = None
    image_path_in_db = None
    pdf_path_in_db = None
    pdf_documents = []
    use_pdf_only = False

    # PDF 처리
    if pdfs:
        pdf_path = pdfs[0].path
        saved_path, error = save_pdf_file(pdf_path, user_name)

        if error:
            error_msg = cl.Message(content=f"❌ PDF 저장 중 오류가 발생했습니다:\n{error}", author="science_chatbot")
            await error_msg.send()
        else:
            pdf_path_in_db = saved_path
            pdf_filename = os.path.basename(pdf_path_in_db)

            pdf_documents, error = extract_text_from_pdf_with_opendataloader(pdf_path, pdf_filename)

            if error:
                error_msg = cl.Message(content=f"❌ PDF 처리 중 오류가 발생했습니다:\n{error}", author="science_chatbot")
                await error_msg.send()
            else:
                use_pdf_only = True
                uploaded_pdf_docs = pdf_documents
                cl.user_session.set("uploaded_pdf_docs", uploaded_pdf_docs)

                pdf_info_msg = cl.Message(
                    content=f"✅ PDF 파일이 성공적으로 업로드되었습니다 ({len(pdf_documents)}개 청크 추출).\n이 PDF의 내용을 바탕으로 질문에 답변하겠습니다.",
                    author="science_chatbot"
                )
                await pdf_info_msg.send()

    # 이미지 처리
    if images:
        image_path = images[0].path

        # 1. 이미지 파일 영구 저장
        saved_path, error = save_image_file(image_path, user_name)

        if error:
            error_msg = cl.Message(content=f"❌ 이미지 저장 중 오류가 발생했습니다:\n{error}", author="science_chatbot")
            await error_msg.send()
        else:
            image_path_in_db = saved_path

            # 2. 이미지 변환 (멀티모달 LLM용 - HEIC도 JPEG로 변환)
            try:
                jpeg_base64, error = convert_image_to_jpeg(image_path)

                if error:
                    error_msg = cl.Message(content=f"❌ 이미지 처리 중 오류가 발생했습니다:\n{error}", author="science_chatbot")
                    await error_msg.send()
                    image_base64 = None
                else:
                    image_base64 = jpeg_base64

            except Exception as e:
                import traceback
                error_trace = traceback.format_exc()
                error_msg = cl.Message(content=f"❌ 이미지 처리 중 예외가 발생했습니다: {str(e)}\n{error_trace}",
                                       author="science_chatbot")
                await error_msg.send()
                image_base64 = None

    # 문서 검색
    if use_pdf_only == True:
        docs_QNA = []
        docs_PDF = []
    else:
        retriever_QNA = cl.user_session.get("retriever_QNA")
        retriever_PDF = cl.user_session.get("retriever_PDF")
        docs_QNA = retriever_QNA.invoke(message.content)
        docs_PDF = retriever_PDF.invoke(message.content)

    # 컨텍스트 메시지 생성
    if use_pdf_only == True:
        context_messages = create_context_messages(uploaded_pdf_docs=uploaded_pdf_docs)
    else:
        context_messages = create_context_messages(docs_QNA, docs_PDF, uploaded_pdf_docs)

    # LangChain 0.3 표준: 이미지 포함 HumanMessage 생성
    human_message_content = []

    # 이미지가 있으면 먼저 추가
    if image_base64:
        human_message_content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{image_base64}"
            }
        })

    # 텍스트 추가
    human_message_content.append({
        "type": "text",
        "text": message.content
    })

    current_human_message = HumanMessage(content=human_message_content)

    # 메모리에 저장
    messages.append(current_human_message)

    # 시스템 프롬프트 선택 - 동적 생성
    system_prompt = create_dynamic_system_prompt(
        user_name=user_name,
        use_pdf_only=use_pdf_only,
        uploaded_pdf_docs=uploaded_pdf_docs
    )

    # 프롬프트 구성
    prompt = ChatPromptTemplate.from_messages([
        system_prompt,
        *context_messages,
        MessagesPlaceholder(variable_name="chat_history"),
        MessagesPlaceholder(variable_name="input"),
    ])

    base_chain = prompt | llm_local_server | StrOutputParser()

    get_session_history = cl.user_session.get("get_session_history")
    session_id = cl.user_session.get("session_id")

    chain_with_history = RunnableWithMessageHistory(
        base_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )

    # 응답 생성 및 스트리밍
    msg = cl.Message(content="", author="science_chatbot")
    await msg.send()

    collected_output = ""
    start_time = time.time()

    try:
        user_input_for_chain = [current_human_message]

        async for chunk in chain_with_history.astream(
                {"input": user_input_for_chain},
                config={"configurable": {"session_id": session_id}}
        ):
            # 1. chunk가 객체(AIMessageChunk)인지 문자열인지 확인하여 원본 콘텐츠 추출
            if hasattr(chunk, "content"):
                raw_content = chunk.content
            else:
                raw_content = chunk

            # 2. Gemini-3 등에서 발생하는 리스트 형태의 콘텐츠 처리 로직 적용
            text_content = ""
            if isinstance(raw_content, list):
                # 리스트인 경우 텍스트 부분만 추출하여 결합
                text_content = ''.join([
                    item.get('text', '') if isinstance(item, dict) else str(item)
                    for item in raw_content
                ])
            elif isinstance(raw_content, str):
                text_content = raw_content
            else:
                # 그 외 타입은 문자열로 강제 변환
                text_content = str(raw_content)

            # 3. 빈 문자열이 아닐 경우에만 출력 및 저장
            if text_content:
                collected_output += text_content
                await msg.stream_token(text_content)

            elapsed_time = time.time() - start_time
            if elapsed_time > 60:
                raise asyncio.TimeoutError("문장 생성 시간이 60초가 지나 문장 생성을 중단합니다.")

        # 응답 후 메시지 업데이트
        messages.append(AIMessage(content=collected_output))
        cl.user_session.set("messages", messages)

    except asyncio.TimeoutError as e:
        if collected_output:
            messages.append(AIMessage(content=collected_output))
            cl.user_session.set("messages", messages)
        await msg.stream_token("\n\n[시간 초과로 응답이 중단되었습니다]")

    except Exception as e:
        import traceback
        print(f"상세 오류: {traceback.format_exc()}")
        await msg.stream_token(f"\n\n[오류가 발생했습니다: {str(e)}]")

    await msg.update()

    session_id = cl.user_session.get("id")
    accuracy = 0
    satisfaction = 0

    connection = pymysql.connect(
        host=os.getenv('DB_HOST'),
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD'),
        db=os.getenv('DB_NAME'),
        charset='utf8mb4',
        cursorclass=pymysql.cursors.DictCursor
    )

    try:
        with connection.cursor() as cursor:
            sql = """INSERT INTO datalog_gen_3
                     (user_name, session_id, student_question, answer, accuracy, satisfaction, image_path, pdf_path, \
                      selected_similarity)
                     VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)"""

            cursor.execute(sql, (
                user_name,
                session_id,
                message.content,
                collected_output,
                accuracy,
                satisfaction,
                image_path_in_db if image_path_in_db else None,
                pdf_path_in_db if pdf_path_in_db else None,
                None
            ))
        connection.commit()
    finally:
        connection.close()

    # 평가 버튼 표시
    actions = [
        cl.Action(name="correct_btn", payload={"value": "correct"}, label="✅ 정확한 설명"),
        cl.Action(name="wrong_btn", payload={"value": "wrong"}, label="❌ 틀린 설명"),
        cl.Action(name="satisfied_btn", payload={"value": "satisfied"}, label="👍 적당한 수준의 설명"),
        cl.Action(name="dissatisfied_btn", payload={"value": "dissatisfied"}, label="👎 너무 쉽거나 어려운 설명")
    ]
    await cl.Message(content="답변의 정확성과 답변의 설명 수준 적절성을 평가해 주세요", actions=actions, author="science_chatbot").send()

    # 유사 질문-답변 쌍을 Action으로 제시
    if not use_pdf_only:
        # 1. 높은 유사도로 필터링된 질문 (0.5 이상, 최대 2개)
        docs_high_similarity = vectorstore_QNA.similarity_search_with_relevance_scores(
            message.content,
            k=10  # 충분히 많이 가져온 후 필터링
        )

        high_similarity_actions = []
        for doc, score in docs_high_similarity:
            if score >= 0.5 and len(high_similarity_actions) < 2:  # 0.5 이상, 최대 2개
                if doc.metadata.get("답변"):
                    question = doc.page_content
                    answer = doc.metadata.get("답변")
                    payload = {
                        "question": question,
                        "answer": answer,
                        "similarity": float(score)
                    }
                    high_similarity_actions.append(
                        cl.Action(
                            name="similar_question",
                            payload=payload,
                            label=f"❓ [유사도: {score:.2f}] {question[:80]}"
                        )
                    )

        # 2. MMR 검색 (항시 4개, 기존 설정 유지)
        docs_QNA = vectorstore_QNA.max_marginal_relevance_search(
            message.content,
            k=4,
            fetch_k=40,
            lambda_mult=0.95  # 기존 최적값 유지
        )

        mmr_actions = []
        for doc in docs_QNA:
            if doc.metadata.get("답변"):
                question = doc.page_content
                answer = doc.metadata.get("답변")
                payload = {
                    "question": question,
                    "answer": answer,
                    "similarity": 0  # MMR은 점수 없음
                }
                mmr_actions.append(
                    cl.Action(
                        name="similar_question",
                        payload=payload,
                        label=f"❓ {question[:80]}"
                    )
                )

        # 3. 메시지 표시
        if high_similarity_actions:
            await cl.Message(
                content="💡 직접 관련된 질문들:",
                actions=high_similarity_actions,
                author="science_chatbot"
            ).send()

        if mmr_actions:
            await cl.Message(
                content="🔍 다양한 관점의 질문들:",
                actions=mmr_actions,
                author="science_chatbot"
            ).send()

@cl.action_callback("similar_question")
async def on_similar_question(action: cl.Action):
    """
    유사 질문 Action을 클릭했을 때 답변을 보여주고 DB에 로그를 남깁니다.
    """
    payload = action.payload
    question = payload.get("question")
    answer = payload.get("answer")
    similarity = payload.get("similarity")

    await cl.Message(content=f"**선택한 질문:**\n{question}\n\n**답변:**\n{answer}", author="science_chatbot").send()

    user_name = cl.user_session.get("user_name", "미등록")
    session_id = cl.user_session.get("id")

    connection = None
    try:
        connection = pymysql.connect(
            host=os.getenv('DB_HOST'),
            user=os.getenv('DB_USER'),
            password=os.getenv('DB_PASSWORD'),
            db=os.getenv('DB_NAME'),
            charset='utf8mb4',
            cursorclass=pymysql.cursors.DictCursor
        )
        with connection.cursor() as cursor:
            sql = """INSERT INTO datalog_gen_3
                     (user_name, session_id, student_question, answer, accuracy, satisfaction, selected_similarity)
                     VALUES (%s, %s, %s, %s, %s, %s, %s)"""
            cursor.execute(sql, (user_name, session_id, question, answer, 0, 0, float(similarity)))
        connection.commit()
    except Exception as e:
        print(f"DB Error on similar question logging: {e}")
        await cl.Message(content=f"DB 관련 오류가 발생했습니다: {e}", author="science_chatbot").send()
    finally:
        if connection:
            connection.close()


@cl.action_callback("correct_btn")
async def on_correct(action):
    accuracy = 2
    session_id = cl.user_session.get("id")
    connection = None
    try:
        connection = pymysql.connect(host=os.getenv('DB_HOST'), user=os.getenv('DB_USER'),
                                     password=os.getenv('DB_PASSWORD'), db=os.getenv('DB_NAME'),
                                     charset='utf8mb4', cursorclass=pymysql.cursors.DictCursor)
        with connection.cursor() as cursor:
            cursor.execute("SELECT MAX(input_time) as max_time FROM datalogWHERE session_id = %s", (session_id,))
            result = cursor.fetchone()
            if result and result['max_time']:
                max_time = result['max_time']
                sql = "UPDATE datalogSET accuracy = %s WHERE session_id = %s AND input_time = %s"
                cursor.execute(sql, (accuracy, session_id, max_time))
        connection.commit()
    finally:
        if connection:
            connection.close()
    await cl.Message(content="답변이 정확했다고 등록했습니다.", author="science_chatbot").send()


@cl.action_callback("wrong_btn")
async def on_wrong(action):
    accuracy = 1
    session_id = cl.user_session.get("id")
    connection = None
    try:
        connection = pymysql.connect(host=os.getenv('DB_HOST'), user=os.getenv('DB_USER'),
                                     password=os.getenv('DB_PASSWORD'), db=os.getenv('DB_NAME'),
                                     charset='utf8mb4', cursorclass=pymysql.cursors.DictCursor)
        with connection.cursor() as cursor:
            cursor.execute("SELECT MAX(input_time) as max_time FROM datalogWHERE session_id = %s", (session_id,))
            result = cursor.fetchone()
            if result and result['max_time']:
                max_time = result['max_time']
                sql = "UPDATE datalogSET accuracy = %s WHERE session_id = %s AND input_time = %s"
                cursor.execute(sql, (accuracy, session_id, max_time))
        connection.commit()
    finally:
        if connection:
            connection.close()
    await cl.Message(content="답변이 틀렸다고 등록했습니다.", author="science_chatbot").send()


@cl.action_callback("satisfied_btn")
async def on_accurate(action):
    satisfaction = 2
    session_id = cl.user_session.get("id")
    connection = None
    try:
        connection = pymysql.connect(host=os.getenv('DB_HOST'), user=os.getenv('DB_USER'),
                                     password=os.getenv('DB_PASSWORD'), db=os.getenv('DB_NAME'),
                                     charset='utf8mb4', cursorclass=pymysql.cursors.DictCursor)
        with connection.cursor() as cursor:
            cursor.execute("SELECT MAX(input_time) as max_time FROM datalogWHERE session_id = %s", (session_id,))
            result = cursor.fetchone()
            if result and result['max_time']:
                max_time = result['max_time']
                sql = "UPDATE datalogSET satisfaction = %s WHERE session_id = %s AND input_time = %s"
                cursor.execute(sql, (satisfaction, session_id, max_time))
        connection.commit()
    finally:
        if connection:
            connection.close()
    await cl.Message(content="답변의 설명 수준이 적당했다고 등록했습니다.", author="science_chatbot").send()


@cl.action_callback("dissatisfied_btn")
async def on_not_accurate(action):
    satisfaction = 1
    session_id = cl.user_session.get("id")
    connection = None
    try:
        connection = pymysql.connect(host=os.getenv('DB_HOST'), user=os.getenv('DB_USER'),
                                     password=os.getenv('DB_PASSWORD'), db=os.getenv('DB_NAME'),
                                     charset='utf8mb4', cursorclass=pymysql.cursors.DictCursor)
        with connection.cursor() as cursor:
            cursor.execute("SELECT MAX(input_time) as max_time FROM datalogWHERE session_id = %s", (session_id,))
            result = cursor.fetchone()
            if result and result['max_time']:
                max_time = result['max_time']
                sql = "UPDATE datalogSET satisfaction = %s WHERE session_id = %s AND input_time = %s"
                cursor.execute(sql, (satisfaction, session_id, max_time))
        connection.commit()
    finally:
        if connection:
            connection.close()
    await cl.Message(content="답변의 설명 수준이 너무 쉽거나 어렵다고 등록했습니다.", author="science_chatbot").send()


if __name__ == "__main__":
    cl.run()
