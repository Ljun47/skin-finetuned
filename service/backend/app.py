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
# 💡 분리된 핵심 모듈 임포트
# --------------------------
from constants import (
    SYSTEM_PROMPT,
    OFF_TOPIC_REFUSAL_KO,
    OFF_TOPIC_REFUSAL_EN,
    IMAGE_VETTING_REFUSAL_KO,
    IMAGE_VETTING_REFUSAL_EN,
    MAX_PROMPT_LENGTH_CHAR
)
from translator import GeminiTranslator
from inference_client import LlavaInferenceClient

# --------------------------
# 💡 전역 서비스 인스턴스 초기화
# --------------------------
# React 빌드 경로 매핑
FRONTEND_BUILD_PATH = os.getenv("FRONTEND_BUILD_PATH", os.path.join(os.path.dirname(__file__), '../frontend/dist'))

# 번역기 및 추론 클라이언트 초기화
translator = GeminiTranslator()
inference_client = LlavaInferenceClient()

# Flask 앱 초기화
app = Flask(__name__, static_folder=FRONTEND_BUILD_PATH)
CORS(app)

# --------------------------
# 💡 루트 경로('/') 및 정적 파일 처리
# --------------------------
@app.route('/', methods=['GET'])
def serve_frontend():
    try:
        return send_from_directory(app.static_folder, 'index.html')
    except FileNotFoundError:
        return jsonify({
            "error": f"Frontend build files not found in path: {FRONTEND_BUILD_PATH}. Check your path and build status."
        }), 500

@app.route('/<path:path>')
def serve_static_files(path):
    return send_from_directory(app.static_folder, path)

# --------------------------
# 💡 핵심 2: 추론 API (/predict)
# --------------------------
@app.route('/predict', methods=['POST'])
def predict():
    # 1. 초기 유효성 검사
    if not inference_client.is_available():
        return jsonify({"error": "Hugging Face Endpoint URL이 설정되지 않았습니다."}), 500

    if not translator.is_available():
        return jsonify({"error": "Gemini API 키가 설정되지 않아 번역 기능을 사용할 수 없습니다."}), 503

    try:
        data = request.get_json()
        if not data or 'text_input' not in data or 'image_data' not in data or 'chat_history' not in data:
            return jsonify({"error": "요청에 'text_input', 'image_data', 또는 'chat_history'가 누락되었습니다."}), 400

        user_prompt_ko = data['text_input']
        base64_image_data_url = data['image_data']
        chat_history = data['chat_history']

        # 2. 사용자 입력 번역 (한국어 -> 영어)
        user_prompt_en = translator.translate(user_prompt_ko, target_language='English')

        # 3. Base64 이미지 데이터 정규화
        image_url_for_endpoint = ""
        if base64_image_data_url:
            if base64_image_data_url.startswith('data:image'):
                base64_image = base64_image_data_url.split(',', 1)[-1]
            else:
                base64_image = base64_image_data_url
            image_url_for_endpoint = f"data:image/jpeg;base64,{base64_image}"

        # 4. LLaVA 멀티턴 프롬프트 형식 구성 (번역된 텍스트 사용)
        final_prompt = SYSTEM_PROMPT

        # 컨텍스트 초과 오류 방지를 위해 최근 3턴만 프롬프트에 병합
        chat_history_limited = chat_history[-3:]

        for i, message in enumerate(chat_history_limited):
            translated_text = translator.translate(message['text'], target_language='English')
            role_prefix = "USER: " if message['role'] == 'user' else "ASSISTANT: "

            # 첫 번째 턴의 사용자 입력에만 이미지 토큰(<image>) 포함
            if message['role'] == 'user' and i == 0:
                final_prompt += f"USER: <image>\n{translated_text} "
            else:
                final_prompt += f"{role_prefix}{translated_text} "

        # 현재 사용자의 신규 질문(번역된 영어)을 최종 조립
        final_prompt += f"USER: {user_prompt_en} ASSISTANT:"

        # 5. 컨텍스트 길이 사전 체크 (Safety Check)
        if len(final_prompt) > MAX_PROMPT_LENGTH_CHAR:
            error_msg_ko = translator.translate(
                "대화가 길어져서 컨텍스트 제한을 초과했습니다. 새로운 대화를 다시 열어 주세요.",
                target_language='Korean',
                source_language='English'
            )
            return jsonify({"error": error_msg_ko}), 413

        # 6. LLaVA Endpoint 추론 호출
        try:
            assistant_response_en = inference_client.query_dermatology_model(
                prompt=final_prompt,
                base64_image_data=image_url_for_endpoint
            )
        except requests.exceptions.HTTPError as http_err:
            error_msg_ko = translator.translate(
                str(http_err), 
                target_language='Korean', 
                source_language='English'
            )
            return jsonify({"error": error_msg_ko}), http_err.response.status_code

        # 7. LLaVA 가드레일 판단 및 응답 덮어쓰기
        save_to_history_flag = True
        assistant_response_ko = ""

        if OFF_TOPIC_REFUSAL_EN in assistant_response_en:
            # 비전문 도메인 거부 처리
            assistant_response_ko = OFF_TOPIC_REFUSAL_KO
            print(f"🛑 LLaVA 가드레일 감지: 비전문 분야 차단. 질문: '{user_prompt_ko}'")
            save_to_history_flag = False
        elif IMAGE_VETTING_REFUSAL_EN in assistant_response_en:
            # 이미지 품질 미달 거부 처리
            assistant_response_ko = IMAGE_VETTING_REFUSAL_KO
            print(f"🛑 LLaVA 가드레일 감지: 이미지 품질 불합격. 질문: '{user_prompt_ko}'")
            save_to_history_flag = False
        else:
            # 8. 최종 응답 번역 (영어 -> 한국어)
            assistant_response_ko = translator.translate(
                assistant_response_en, 
                target_language='Korean', 
                source_language='English'
            )

        result = {
            "input": user_prompt_ko,
            "generated_text": assistant_response_ko,
            "save_to_history": save_to_history_flag
        }
        return jsonify(result)

    except Exception as e:
        print(f"Prediction Error (Internal Exception): {e}")
        error_message_ko = translator.translate(
            f"Flask 내부 서버 오류: {str(e)}", 
            target_language='Korean', 
            source_language='English'
        )
        return jsonify({"error": error_message_ko}), 500

# --------------------------
# 💡 서버 기동 진입점
# --------------------------
if __name__ == "__main__":
    PORT = int(os.getenv("PORT", 5000))
    NGROK_AUTH_TOKEN = os.getenv("NGROK_AUTH_TOKEN", "")
    
    public_url = None
    if NGROK_AUTH_TOKEN:
        try:
            from pyngrok import ngrok
            ngrok.set_auth_token(NGROK_AUTH_TOKEN)
            ngrok.kill()
            public_url = ngrok.connect(PORT).public_url
            print(f"🎉 Public Tunnel URL (ngrok): {public_url}")
            print("이 URL을 프론트엔드 API 호출 경로로 사용하십시오.")
        except Exception as e:
            print(f"ngrok 연결 실패: {e}. 로컬 기동으로 계속 진행합니다.")

    print(f"Starting Flask server on port {PORT}...")
    http_server = WSGIServer(('0.0.0.0', PORT), app)
    http_server.serve_forever()
