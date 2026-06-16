# 🩺 방구석피부과

> **한국인 안면 피부질환 자가 진단을 위한 Multimodal LLM 파인튜닝**

본 프로젝트는 사용자가 제공한 얼굴 피부 사진과 텍스트 질의를 바탕으로 안면 피부질환을 진단하고, 증상에 적합한 가이드라인 및 전문적인 지식 설명을 제공하는 멀티모달 AI 시스템입니다.

---

## 🌟 주요 기능 (Features)

*   **Multimodal Fine-Tuning**: LLaVA-1.5-7B 모델에 한국인 피부질환 데이터를 학습시켜 안면 질환 진단 능력 제고.
*   **Context-Augmented Alignment (RAFT)**: RAG 파이프라인 상의 정보 추출(Citation) 정확도와 노이즈 지문 차단력 개선을 위한 미세 조정 수행.
*   **Dual-Translation Pipeline**: Gemini-2.5-Flash API를 결합하여 한국어 입력을 영어로 가공 후 LLaVA 추론을 거쳐 최종 한국어로 응답을 복원하는 번역 아키텍처.
*   **Input & Domain Guardrails**: 
    *   입력 이미지 해상도/조도 유효성 검수 및 비전 검사 (Vetting).
    *   피부 건강 이외의 주제에 대한 질의 차단 필터 (Off-Topic Refusal).

---

## 🏗️ 시스템 아키텍처 (System Architecture)

```mermaid
graph TD
    User([사용자]) -->|1. 한국어 질문 & 이미지| FE[React Frontend]
    FE -->|2. POST /predict| BE[Flask Backend]
    
    subgraph Translation & Vetting (Gemini API)
        BE -->|3. 번역 및 가드레일 검수| Gemini[Gemini-2.5-Flash]
        Gemini -->|4. 번역 결과 및 안전성 판정| BE
    end

    subgraph Core Inference (Hugging Face Endpoint)
        BE -->|5. 영어 질문 + 이미지| HF[LLaVA-1.5-7B SFT Model]
        HF -->|6. 영어 추론 결과| BE
    end

    subgraph Output Translation
        BE -->|7. 영-한 번역 요청| Gemini
        Gemini -->|8. 최종 한국어 응답| BE
    end

    BE -->|9. 진단 결과 응답| FE
    FE -->|10. 화면 표출| User
```

---

## 📈 학습 정보 (Training Details)

### 1. Stage 1: SFT (Supervised Fine-Tuning)
*   **Base Model**: `llava-hf/llava-1.5-7b-hf`
*   **Dataset**: AI Hub 한국인 안면 피부질환 이미지 9,600장 기반 멀티턴 대화쌍 38,400 Turn.
*   **Training Config**: QLoRA (8-bit Quantization), $R=16$, $\alpha=32$, Learning Rate $1\times10^{-4}$, Epoch 1 (조기 종료).
*   **Metrics Evaluation**:
    *   **Accuracy**: Base 모델 대비 약 **60%** 향상 (0.093 ➔ 0.148)
    *   **Macro F1-Score**: Base 모델 대비 약 **65%** 향상 (0.126 ➔ 0.208)

### 2. Stage 2: RAFT (Retrieval-Augmented Fine-Tuning)
*   **Dataset**: 여드름, 아토피 등 5개 질환군 RAG 지식 지문 4,240 세트 (Oracle Document 1, Hard Negatives 3, Easy Negative 1 혼합 셔플).
*   **Prompt Configuration**: 컨텍스트 지문 중 올바른 근거 번호를 식별하여 명시하도록 타겟 포맷 강제 (`근거 문맥: [번호]\n답변: {Answer}`).
*   **Technical Note on RAFT Training**:
    *   2차 RAG 얼라인먼트 학습 시, 텍스트 전용(Text-only) 지문 위주로 파인튜닝을 진행하는 과정에서 모델의 비전-언어 결합 가중치 정렬에 간섭이 발생하는 **Modality Mismatch** 현상이 발생하여 Train Loss가 다소 요동치는 경향을 보였습니다. 
    *   향후 고도화를 위해 학습 시 이미지 픽셀 정보(pixel_values)를 누락하지 않는 멀티모달 형태의 RAG-RAFT 기법 도입을 고려 중입니다.

---

## 📂 프로젝트 구조 (Repository Structure)

대용량 이미지 데이터셋의 원격 저장소 노출을 방지하기 위해 원본 데이터는 `.gitignore` 처리되었으며, 데이터 규격 검토용 **50라인의 샘플 스키마 데이터**가 저장소에 탑재되어 있습니다.

```
├── README.md                       # 프로젝트 소개 및 실행 문서
├── data/
│   ├── sample_data/                # 데이터 구조 파악을 위한 샘플 스키마 파일
│   │   ├── sample_sft.jsonl        # 1차 SFT 데이터셋 예시 (50 line)
│   │   └── sample_raft_raw.jsonl   # 2차 RAFT 데이터셋 예시 (50 line)
│   └── scripts/
│       └── merge_knowledge.py      # TF-IDF 기반 지식 전처리 및 2차 데이터셋 병합
├── training/
│   ├── stage1-sft/
│   │   └── train_sft.py            # LLaVA 8-bit QLoRA 1차 SFT 학습 파이프라인
│   └── stage2-raft/
│       └── train_raft.py           # 2차 RAFT RAG-SFT 학습 파이프라인
└── service/
    ├── backend/
    │   ├── app.py                  # Flask 웹 서버 메인 (Gemini & Endpoint 인터페이스)
    │   ├── requirements.txt        # 백엔드 패키지 사양
    │   ├── .env.example            # 로컬 환경변수 템플릿
    │   └── .env                    # [Local Only] API 키 정보
    └── frontend/
        └── dist/                   # UI 배포용 React 빌드 아티팩트
```

---

## ⚙️ 실행 방법 (How to Run)

### 1. 환경 변수 설정
프로젝트 루트 또는 `service/backend/` 폴더에 `.env` 파일을 생성하고 아래 규격에 맞게 변수를 선언합니다. (학습용 코드는 상위 디렉토리의 `.env`를 자동으로 스캔합니다.)
```env
# Gemini API Key (번역 및 가드레일 제어용)
GEMINI_API_KEY=your_gemini_api_key

# Hugging Face API Read Token (Inference Endpoint 호출용)
HF_API_TOKEN=your_huggingface_read_token

# Hugging Face API Write Token (Fine-Tuning 모델 업로드용)
HF_TOKEN=your_huggingface_write_token

# ngrok 인증 토큰 (로컬 Flask 서버 터널링용)
NGROK_AUTH_TOKEN=your_ngrok_auth_token

# Hugging Face Inference Endpoint URL
HF_ENDPOINT_URL=your_huggingface_endpoint_url
```

### 2. 패키지 설치 및 실행
```bash
cd service/backend
pip install -r requirements.txt
python app.py
```
