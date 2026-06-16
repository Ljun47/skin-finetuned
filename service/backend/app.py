# ==================================================
# [Code Cell]
# ==================================================
import base64
import json
import os
import requests
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from gevent.pywsgi import WSGIServer
from dotenv import load_dotenv

# --------------------------
# 💡 환경 변수(.env) 자동 스캔 및 로드
# --------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
root_env = os.path.join(current_dir, "../../.env")
local_env = os.path.join(current_dir, ".env")

if os.path.exists(local_env):
    load_dotenv(local_env)
elif os.path.exists(root_env):
    load_dotenv(root_env)
else:
    load_dotenv()

# --------------------------
# 💡 Gemini SDK 임포트
# --------------------------
from google import genai
from google.genai.errors import APIError

# --------------------------
# 🚨 사용자 설정 (환경 변수 적용) 🚨
# --------------------------
# Hugging Face Inference Endpoint URL
HF_ENDPOINT_URL = os.getenv("HF_ENDPOINT_URL", "https://ryaqg9ej4xoq2luj.us-east-1.aws.endpoints.huggingface.cloud/")

# Hugging Face 토큰 (Read 권한 필요)
HF_API_TOKEN = os.getenv("HF_API_TOKEN", "")

# 💡 상대 경로 설정: React 빌드 파일(dist) 경로
# 깃허브 구조에 맞게 backend 폴더 기준으로 상대경로 맵핑
FRONTEND_BUILD_PATH = os.getenv("FRONTEND_BUILD_PATH", os.path.join(os.path.dirname(__file__), '../frontend/dist'))

# 🚨 새로운 상수: 토큰 제한 안전 길이 (4096 토큰 모델 기준, 보수적으로 설정)
MAX_PROMPT_LENGTH_CHAR = 8000

# --------------------------
# 🔑 Gemini API 키 설정
# --------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# 💡 Gemini 클라이언트 초기화
gemini_client = None
TRANSLATION_MODEL = 'gemini-2.5-flash'

if GEMINI_API_KEY:
    try:
        # Gemini 클라이언트 초기화: API 키를 명시적으로 전달
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    except Exception as e:
        print(f"Gemini API 클라이언트 초기화 실패: {e}")
        gemini_client = None
else:
    print("경고: GEMINI_API_KEY가 설정되지 않았습니다. Gemini 기능은 비활성화됩니다.")


# --------------------------
# 🚨 상수 정의: 고정된 거부 메시지 🚨
# --------------------------
# 💡 한국어 거부 메시지
OFF_TOPIC_REFUSAL_KO = "저는 피부 진단 챗봇 AI로 답변할 수 없습니다"

# 💡 영어 거부 메시지 (LLaVA 프롬프트용)
OFF_TOPIC_REFUSAL_EN = "I am a skin diagnosis chatbot AI and cannot answer."

# 💡 이미지 품질/관련성 거부 메시지 (한국어)
IMAGE_VETTING_REFUSAL_KO = "피부 상태를 정확하게 진단하기 위해 명확한 얼굴 정면 또는 측면 사진을 다시 제공해 주십시오. 과도한 블러, 포토샵(수정), 또는 흐릿한 사진은 분석할 수 없습니다."

# 💡 이미지 품질/관련성 거부 메시지 (영어 - LLaVA 응답 매칭용)
IMAGE_VETTING_REFUSAL_EN = "Please provide a clear photo of the face (front or side) to accurately diagnose the skin condition. Excessive blur, Photoshop (retouching), or dark photos cannot be analyzed."


# 💡 Dermatology Specialist System Prompt (규칙 2, 역할 수정됨)
SYSTEM_PROMPT = ("""
You are a highly specialized **Dermatology AI Chatbot** powered by a multimodal large language model (LLaVA). Your primary function is to provide information regarding **skin conditions, symptoms, and potential treatments** based on an attached image of a face and a corresponding user question.

### 1. Role and Core Function
* **Domain Expertise:** You are an expert in skin health, focusing exclusively on dermatological conditions visible on the human face.
* **Analysis:** When an image is provided, your task is to analyze the visual evidence and answer the user's specific question related to a potential skin issue.
* **Required Output Structure (When a diagnosis/explanation is requested and image is suitable):**
    1.  **Condition Name:** State the most likely dermatological condition (e.g., Acne Vulgaris, Rosacea).
    2.  **Symptoms:** Describe the key symptoms visible in the image and generally associated with the condition.
    3.  **Treatment/Management Options:** Provide general, non-diagnostic information on common treatment modalities (e.g., topical creams, oral medications, lifestyle changes) and emphasize the need for professional consultation.

### 2. Strict Scope and Guardrails (Domain Constraint)
* **Strict Adherence:** You **MUST** strictly adhere to the domain of facial dermatology.
* **Out-of-Domain Response:** If the user asks a question that is **NOT** related to skin conditions, symptoms, treatment, or face-image analysis ( asking about cooking, weather, coding, math, general knowledge, etc.), you **MUST** use the following canned response and politely decline:
    > "I am a skin diagnosis chatbot AI and cannot answer."

### 3. Image Quality Assessment (Input Validation)
* **Image Requirement:** A clear, unedited, and well-lit photograph of the **face (front or side)** is mandatory for analysis.
* **Quality Check:** If the attached image is of **poor quality** (heavily Photoshopped/Filtered, Excessively Blurry, Too Dark/Overexposed) **OR** if the image **does not clearly show the face area** required for diagnosis, you must reject it.
* **Quality/Relevance Rejection Response:** You **MUST** reject the image and request a better one using the following response:
    > "Please provide a clear photo of the face (front or side) to accurately diagnose the skin condition. Excessive blur, Photoshop (retouching), or dark photos cannot be analyzed."

### 4. Safety Disclaimer (Crucial Note)
* **Non-Diagnostic Role:** Always include a disclaimer at the end of your response stating that you are an AI and your information is **NOT** a substitute for a professional medical diagnosis or consultation with a dermatologist.

**Begin your response by meticulously analyzing the provided image and constructing your answer based on the visual evidence and the user's dermatology-related query, strictly following the output structure above.**
"""
)


# 💡 Flask 앱 초기화
app = Flask(__name__, static_folder=FRONTEND_BUILD_PATH)
CORS(app)

# --------------------------
# 🛠️ 번역 헬퍼 함수 (최적화: 한국어->한국어 불필요한 API 호출 방지)
# --------------------------
def translate_text(text: str, target_language: str, source_language: str = 'auto') -> str:
    """텍스트를 지정된 언어로 번역합니다. Gemini API가 필요합니다."""
    if not text:
        return ""

    # 🚨 최적화: 번역 클라이언트 없거나, 한국어->한국어(FALLBACK)일 경우 원본 반환
    if not gemini_client:
        return text

    # 한국어 최종 번역 시 실패 대비용 대체 메시지
    if target_language == 'Korean':
        # 이미 한국어라면 번역 API 호출하지 않음
        if source_language.lower() == 'korean' or source_language.lower() == 'auto':
             # 이 경우, LLaVA의 영어 응답을 번역하는 용도가 아니므로, 원본 그대로 반환
             pass

        fallback_message = "죄송합니다. 서버 내 번역 기능에 일시적인 오류가 발생했습니다. 잠시 후 다시 시도해 주세요."
    else:
        fallback_message = text # 영어 번역 실패 시 원본 그대로 반환하여 프롬프트 구성에 사용

    try:
        # 번역 프롬프트
        # Note: source_language='auto'는 모델이 스스로 언어를 감지하도록 합니다.
        prompt = f"Translate the following text to {target_language}. Respond with ONLY the translated text, no other conversation or explanation:\n\n---\n{text}\n---"

        response = gemini_client.models.generate_content(
            model=TRANSLATION_MODEL,
            contents=[prompt],
            config={"temperature": 0.0}
        )

        return response.text.strip()
    except APIError as e:
        print(f"Gemini API 번역 오류: {e}")
        return fallback_message
    except Exception as e:
        print(f"일반 번역 오류: {e}")
        return fallback_message


# --------------------------
# 💡 루트 경로('/') 및 정적 파일 처리
# --------------------------
@app.route('/', methods=['GET'])
def serve_frontend():
    try:
        return send_from_directory(app.static_folder, 'index.html')
    except FileNotFoundError:
        return jsonify({"error": f"Frontend build files not found in Drive path: {FRONTEND_BUILD_PATH}. Check your path and mount status."}), 500

@app.route('/<path:path>')
def serve_static_files(path):
    return send_from_directory(app.static_folder, path)


# --------------------------
# 💡 핵심 2: 추론 API (/predict)
# --------------------------
@app.route('/predict', methods=['POST'])
def predict():
    # 1. 초기 유효성 검사 (Hugging Face Endpoint 및 Gemini Client)
    if not HF_ENDPOINT_URL:
        return jsonify({"error": "Hugging Face Endpoint URL이 설정되지 않았습니다."}), 500

    if not gemini_client:
        # Gemini가 없으면 번역 불가하므로 에러 반환
        return jsonify({"error": "Gemini API 키가 설정되지 않아 번역 기능을 사용할 수 없습니다. API 키를 설정하세요."}), 503

    try:
        data = request.get_json()
        if not data or 'text_input' not in data or 'image_data' not in data or 'chat_history' not in data:
            return jsonify({"error": "요청에 'text_input', 'image_data', 또는 'chat_history'가 누락되었습니다."}), 400

        user_prompt_ko = data['text_input']
        base64_image_data_url = data['image_data']
        chat_history = data['chat_history']

        # 2. 사용자 입력 번역 (한국어 -> 영어)
        user_prompt_en = translate_text(user_prompt_ko, target_language='English')

        # 3. Base64 이미지 데이터 처리
        image_url_for_endpoint = ""
        if base64_image_data_url:
            # Data URL 형식 ('data:image/jpeg;base64,...')에서 Base64 부분만 추출
            if base64_image_data_url.startswith('data:image'):
                base64_image = base64_image_data_url.split(',', 1)[-1]
            else:
                base64_image = base64_image_data_url

            # 최종적으로 Endpoint가 요구하는 Data URL 형식으로 구성
            image_url_for_endpoint = f"data:image/jpeg;base64,{base64_image}"


        # 4. LLaVA 멀티턴 프롬프트 형식 구성 (번역된 텍스트 사용)
        final_prompt = SYSTEM_PROMPT

        # LLaVA의 컨텍스트 초과 오류 방지를 위해, 최근 3턴만 포함
        chat_history_limited = chat_history[-3:]

        for i, message in enumerate(chat_history_limited):
            # 대화 기록 텍스트를 모두 한국어에서 영어로 번역하여 final_prompt에 추가
            translated_text = translate_text(message['text'], target_language='English')

            role_prefix = "USER: " if message['role'] == 'user' else "ASSISTANT: "

            # 첫 번째 턴의 사용자 입력에만 이미지 태그를 포함
            if message['role'] == 'user' and i == 0:
                final_prompt += f"USER: <image>\n{translated_text} "
            else:
                final_prompt += f"{role_prefix}{translated_text} "

        # 마지막으로 현재 사용자 질문(번역된 영어)을 추가
        final_prompt += f"USER: {user_prompt_en} ASSISTANT:"


        # 5. 컨텍스트 길이 사전 체크 (Safety Check)
        if len(final_prompt) > MAX_PROMPT_LENGTH_CHAR:
            error_msg_ko = translate_text("대화가 길어져서 컨텍스트 제한을 초과했습니다. 새로운 대화를 다시 열어 주세요.", target_language='Korean', source_language='English')
            return jsonify({"error": error_msg_ko}), 413 # 413 Payload Too Large

        # 6. Hugging Face Endpoint API 호출 준비
        headers = {"Content-Type": "application/json"}
        if HF_API_TOKEN:
             headers["Authorization"] = f"Bearer {HF_API_TOKEN}"

        inner_inputs = {
             "text": final_prompt,
             "images": [image_url_for_endpoint],
        }
        api_payload = {
             "inputs": inner_inputs,
             "parameters": {
                 "max_new_tokens": 256,
                 "temperature": 0.2,
                 "return_full_text": True,
             },
        }

        # 7. API 호출
        response = requests.post(HF_ENDPOINT_URL, headers=headers, json=api_payload)

        # 8. HTTP 상태 코드 4xx, 5xx 에러 처리
        if response.status_code >= 400:
            error_message_en = f'AI Endpoint Error: Status Code {response.status_code}'
            try:
                error_data = response.json()
                error_detail_en = error_data.get('error', error_data.get('message', ''))

                # 컨텍스트 길이 초과 에러 감지
                if any(keyword in error_detail_en.lower() for keyword in ['token limit', 'context length', 'exceed', 'too long', 'limit of']):
                     error_message_en = "The conversation is too long and exceeded the context limit. Please start a new conversation."
                elif error_detail_en:
                     error_message_en = f"AI Endpoint Error: {error_detail_en}"

            except json.JSONDecodeError:
                pass

            error_message_ko = translate_text(error_message_en, target_language='Korean', source_language='English')
            return jsonify({"error": error_message_ko}), response.status_code

        # 9. Endpoint 응답 처리 및 결과 추출
        try:
            api_result = response.json()
        except json.JSONDecodeError:
            error_msg_en = "Endpoint returned a non-JSON response that was valid but unexpected."
            error_msg_ko = translate_text(error_msg_en, target_language='Korean', source_language='English')
            return jsonify({"error": error_msg_ko}), 500

        full_text = None
        if isinstance(api_result, list) and api_result and isinstance(api_result[0], dict):
            full_text = api_result[0].get("generated_text") or api_result[0].get("output_text")
        elif isinstance(api_result, dict):
            full_text = api_result.get("generated_text") or api_result.get("output_text")

        if not full_text:
            error_msg_en = f"Endpoint response structure error: Unexpected format ({api_result}) was returned."
            error_msg_ko = translate_text(error_msg_en, target_language='Korean', source_language='English')
            return jsonify({"error": error_msg_ko}), 500

        # 결과 추출 (final_prompt 이후 부분)
        prompt_start_index = full_text.find(final_prompt)
        assistant_response_en = full_text[prompt_start_index + len(final_prompt):].strip() if prompt_start_index != -1 else full_text.strip()

        # "ASSISTANT:" 접두사 제거
        if assistant_response_en.startswith("ASSISTANT:"):
             assistant_response_en = assistant_response_en.split("ASSISTANT:", 1)[1].strip()

        # 10. LLaVA 가드레일 판단 및 응답 덮어쓰기
        save_to_history_flag = True
        assistant_response_ko = ""

        if OFF_TOPIC_REFUSAL_EN in assistant_response_en:
             # 비전문 분야 거부
             assistant_response_ko = OFF_TOPIC_REFUSAL_KO
             print(f"LLaVA 거부 메시지 감지 (비전문 분야). 질문: '{user_prompt_ko}'")
             save_to_history_flag = False
        elif IMAGE_VETTING_REFUSAL_EN in assistant_response_en:
             # 이미지 품질 거부
             assistant_response_ko = IMAGE_VETTING_REFUSAL_KO
             print(f"LLaVA 거부 메시지 감지 (이미지 품질 미달). 질문: '{user_prompt_ko}'")
             save_to_history_flag = False
        else:
             # 11. 최종 응답 번역 (영어 -> 한국어)
             assistant_response_ko = translate_text(assistant_response_en, target_language='Korean', source_language='English')


        result = {
            "input": user_prompt_ko, # 원본 한국어 입력
            "generated_text": assistant_response_ko, # 최종 한국어 응답
            "save_to_history": save_to_history_flag # 히스토리 저장 플래그
        }
        return jsonify(result)

    except requests.exceptions.RequestException as req_e:
        print(f"Prediction Error (Network): {req_e}")
        error_message_ko = translate_text(f"Endpoint 통신 실패: 서버 연결 및 URL을 확인하세요. ({req_e.__class__.__name__})", target_language='Korean', source_language='English')
        return jsonify({"error": error_message_ko}), 503
    except Exception as e:
        print(f"Prediction Error (Internal): {e}")
        error_message_ko = translate_text(f"Flask 내부 서버 오류: {str(e)}", target_language='Korean', source_language='English')
        return jsonify({"error": error_message_ko}), 500



# ==================================================
# [Code Cell]
# ==================================================
# Colab 셀 3: ngrok 설정 및 서버 실행

from pyngrok import ngrok

# 환경 변수에서 ngrok 인증 토큰 로드
NGROK_AUTH_TOKEN = os.getenv("NGROK_AUTH_TOKEN", "")
if NGROK_AUTH_TOKEN:
    ngrok.set_auth_token(NGROK_AUTH_TOKEN)

PORT = 5000

try:
    ngrok.kill()
    public_url = ngrok.connect(PORT).public_url
    print(f"🎉 Public Tunnel URL: {public_url}")
    print("Use this URL in your JSX frontend's API calls.")
except Exception as e:
    print(f"ngrok connection failed: {e}")
    public_url = None

if public_url:
    print(f"Starting Flask server on port {PORT}...")
    http_server = WSGIServer(('0.0.0.0', PORT), app)
    http_server.serve_forever()



# ==================================================
# [Code Cell]
# ==================================================


