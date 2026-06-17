import json
import os
import requests
from typing import Dict, List, Any

class LlavaInferenceClient:
    def __init__(self, endpoint_url: str = None, api_token: str = None):
        self.endpoint_url = endpoint_url or os.getenv(
            "HF_ENDPOINT_URL", 
            "https://ryaqg9ej4xoq2luj.us-east-1.aws.endpoints.huggingface.cloud/"
        )
        self.api_token = api_token or os.getenv("HF_API_TOKEN", "")
        
        if not self.endpoint_url:
            print("⚠️ 경고: HF_ENDPOINT_URL이 구성되지 않았습니다.")

    def is_available(self) -> bool:
        return bool(self.endpoint_url)

    def query_dermatology_model(self, prompt: str, base64_image_data: str, max_new_tokens: int = 256) -> str:
        """
        Hugging Face Inference Endpoint에 요청을 전송하여 LLaVA 추론 결과 반환
        """
        if not self.is_available():
            raise ValueError("Hugging Face Endpoint URL이 설정되지 않았습니다.")

        # API 요청 헤더 작성
        headers = {"Content-Type": "application/json"}
        if self.api_token:
            headers["Authorization"] = f"Bearer {self.api_token}"

        # LLaVA 1.5 Inference Endpoint 표준 입력 페이로드 규격 구성
        inner_inputs = {
            "text": prompt,
            "images": [base64_image_data]
        }
        
        payload = {
            "inputs": inner_inputs,
            "parameters": {
                "max_new_tokens": max_new_tokens,
                "temperature": 0.2,
                "return_full_text": True,
            }
        }

        # POST 요청 전송
        response = requests.post(self.endpoint_url, headers=headers, json=payload, timeout=120)
        
        # HTTP 오류가 발생했을 경우 에러 메시지 매핑
        if response.status_code >= 400:
            error_message = f"AI Endpoint Error: Status Code {response.status_code}"
            try:
                error_data = response.json()
                error_detail = error_data.get("error", error_data.get("message", ""))
                
                # 컨텍스트 초과 에러 문구 감지
                if any(k in error_detail.lower() for k in ["token limit", "context length", "exceed", "too long"]):
                    error_message = "The conversation is too long and exceeded the context limit. Please start a new conversation."
                elif error_detail:
                    error_message = f"AI Endpoint Error: {error_detail}"
            except json.JSONDecodeError:
                pass
                
            raise requests.exceptions.HTTPError(error_message, response=response)

        # JSON 응답 파싱
        try:
            api_result = response.json()
        except json.JSONDecodeError:
            raise ValueError("Endpoint에서 유효하지 않은 응답(비 JSON)이 반환되었습니다.")

        # 반환 형태 파싱 [ {"generated_text": "..."} ] 혹은 {"generated_text": "..."}
        full_text = ""
        if isinstance(api_result, list) and api_result and isinstance(api_result[0], dict):
            full_text = api_result[0].get("generated_text") or api_result[0].get("output_text") or ""
        elif isinstance(api_result, dict):
            full_text = api_result.get("generated_text") or api_result.get("output_text") or ""

        if not full_text:
            raise ValueError(f"예상치 못한 응답 구조가 반환되었습니다. 응답내용: {api_result}")

        # LLaVA 모델의 경우 입력 프롬프트를 포함해서 리턴하므로, 프롬프트 이후의 실질적 답변만 추출
        prompt_start_index = full_text.find(prompt)
        if prompt_start_index != -1:
            assistant_response = full_text[prompt_start_index + len(prompt):].strip()
        else:
            assistant_response = full_text.strip()

        # 접두사 잔재 'ASSISTANT:' 제거
        if assistant_response.startswith("ASSISTANT:"):
            assistant_response = assistant_response.split("ASSISTANT:", 1)[1].strip()

        return assistant_response
