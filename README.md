# RAG & IRQA 과학 질문-답변 챗봇 템플릿

![Python](https://img.shields.io/badge/Python-3.13%2B-blue?logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-Apache%202.0-green)
![Chainlit](https://img.shields.io/badge/UI-Chainlit-F75591?logo=chainlit&logoColor=white)
![RAG](https://img.shields.io/badge/Tech-RAG%20%26%20IRQA-orange)

> **실제 운영 사례 및 관련 학위 논문**
> * **운영 사례 보기**: [https://acer2.snu.ac.kr](https://acer2.snu.ac.kr) (2020년부터 실제 운영 중)
> * **관련 학위논문**: [학위논문 링크](https://s-space.snu.ac.kr/handle/10371/222093?mode=full)


이 프로젝트는 **RAG(검색 증강 생성)** 와 **IRQA(정보 검색 기반 질의응답)** 기술을 결합한 과학 교육용 챗봇 오픈소스 템플릿입니다. 
Chainlit을 기반으로 하여, 사용자의 질문에 대해 **생성형 AI의 답변**과 **데이터베이스 내 가장 유사한 질문-답변** 을 동시에 제공합니다.

다른 개발자들이 이 코드를 기반으로 자신의 데이터(다른 과목의 질의응답 자료, 다른 분야의 질의응답 자료, PDF 파일 내용 등)를 넣어 커스터마이징할 수 있도록 설계되었습니다.


## 주요 특징 (Key Features)

* **LLM과 기존의 질문-답변 데이터셋을 조합한 시스템**:
    * **LLM**: Google Gemini / Ollama가 문맥을 파악하여 친절하게 설명합니다.
    * **기존의 유사 질문-답변 제안**: 벡터 DB에서 질문과 가장 유사한 기존 Q&A(학생들의 질의응답, 학교 정보 등)를 검색하여 원본 데이터를 함께 보여줍니다. 이를 통해 AI 환각(Hallucination)을 교차 검증할 수 있습니다.
* **유연한 LLM 선택**: LLM으로 클라우드(Google Gemini) API와 로컬(Ollama) 모델을 선택할 수 있습니다.
* **멀티모달 지원**: 텍스트뿐만 아니라 이미지, PDF를 업로드하여 내용을 분석하고 질문할 수 있습니다.
* **자동 벡터 저장소 구축**: xlsx 파일이나 PDF 파일을 넣고 스크립트만 실행하면 자동으로 FAISS 벡터 DB를 구축합니다.
* **사용자 만족도 기록**: 사용자의 만족도 피드백을 오픈소스 DB인 MariaDB에 저장하여 추후 모델 개선에 활용합니다.

## 기술 스택

* **프론트엔드**: [Chainlit](https://docs.chainlit.io)
* **LLM**: Google Gemini (Cloud), Ollama (Local)
* **RAG/벡터 저장소**: FAISS (CPU based), LangChain
* **데이터베이스**: MariaDB (사용 기록 및 사용자 평가 저장)
* **자연어처리**: Kiwi (한국어 형태소 분석), HuggingFace Embeddings

---


## 💻 설치 및 실행 (Getting Started)


### 1. 사전 요구 사항
* 운영체제: 리눅스
* Python: 3.13 이상
* 오픈소스 LLM 사용 시 [Ollama](https://ollama.com/) 설치 및 모델 다운로드 (LG AI 연구원의 `EXAONE 3.5 7.8B`나 구글의 `gemma3:12b-it-qat` 이상 권장)
* 데이터베이스: MariaDB


### 2. 설치

* git과 uv가 설치되어 있어야 합니다.

```bash
git clone https://github.com/kungmo/science_qna_rag_irqa_chatbot.git
cd science_qna_rag_irqa_chatbot
uv init --python 3.13
uv pip install -r requirements.txt
```

### 3. 환경 변수 설정 (.env)
프로젝트 루트에 .env 파일을 생성하세요.

```bash
GOOGLE_API_KEY=여기에_Gemini_API_키_입력
DB_HOST=127.0.0.1
DB_USER=여기에_사용자이름_입력
DB_PASSWORD=여기에_비밀번호_입력
DB_NAME=여기에_DB_이름_입력
```


### 4. 데이터베이스 테이블 생성 (SQL)

* DBeaver로 MariaDB데이터베이스에 접근한 후, 아래 SQL을 실행해서 테이블을 생성합니다. (필수)
DB 이름을 chatbot_logs, 테이블 이름을 datalog로 정했다고 가정합니다.

```sql
CREATE DATABASE IF NOT EXISTS `chatbot_logs` DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci;
USE `chatbot_logs`;
CREATE TABLE `datalog` (
  `primarykey` int(11) NOT NULL AUTO_INCREMENT,
  `input_time` datetime NOT NULL DEFAULT current_timestamp(),
  `user_name` varchar(15) NOT NULL,
  `session_id` varchar(36) NOT NULL,
  `student_questio` varchar(2048) DEFAULT NULL,
  `answer` longtext DEFAULT NULL,
  `accuracy` smallint(5) unsigned DEFAULT NULL,
  `satisfaction` smallint(5) unsigned DEFAULT NULL,
  `image_path` varchar(255) DEFAULT NULL,
  `pdf_path` varchar(255) DEFAULT NULL,
  `selected_similar` float DEFAULT NULL,
  PRIMARY KEY (`primarykey`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
```

### 5. 벡터 DB 구축 (나만의 데이터 넣기)

이 챗봇을 **여러분의 목적(과학 질문-답변, 학교 관련 질의응답, 그밖의 질의응답 등)** 에 맞게 바꾸려면 데이터를 교체하세요.

    data/ 폴더:
    - 과학 질문-답변 쌍이 담긴 엑셀 파일(df_qna_*.xlsx)을 넣습니다. (컬럼: 질문, 답변)
    - 과학 외의 질문-답변 쌍이 담긴 엑셀 파일(df_cus_*.xlsx)을 넣습니다. (컬럼: 질문, 답변)

    pdfs/ 폴더:
    - 참고할 PDF 문서를 넣습니다.
    - PDF 여러 개를 넣어도 한꺼번에 처리합니다.

    인덱싱 스크립트 실행:

```bash
python rag_faiss_creator.py           # 엑셀 데이터 인덱싱
python rag_faiss_creator_renew_pdf.py # PDF 데이터 인덱싱
```

### 6. 실행

```Bash
chainlit run chainlit_memory.py
```


### 그밖에...

이 챗봇 틀은 범용적으로 쓸 수 있기 때문에 과학 질의응답이 아닌 다른 영역에도 쓸 수 있습니다.

### 라이선스

Apache License 2.0