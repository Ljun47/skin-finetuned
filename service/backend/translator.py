import os
from google import genai
from google.genai.errors import APIError

class GeminiTranslator:
    def __init__(self, api_key: str = None, model: str = 'gemini-2.5-flash'):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        self.model_name = model
        self.client = None
        
        if self.api_key:
            try:
                # Client 초기화: API 키를 명시적으로 전달
                self.client = genai.Client(api_key=self.api_key)
                print("❇️ Gemini API 번역 클라이언트가 성공적으로 초기화되었습니다.")
            except Exception as e:
                print(f"❌ Gemini API 클라이언트 초기화 실패: {e}")
        else:
            print("⚠️ 경고: GEMINI_API_KEY가 설정되지 않았습니다. 번역 기능이 작동하지 않습니다.")

    def is_available(self) -> bool:
        return self.client is not None

    def translate(self, text: str, target_language: str, source_language: str = 'auto') -> str:
        """
        Gemini 모델을 사용해 텍스트를 지정된 언어로 번역합니다.
        """
        if not text:
            return ""

        if not self.is_available():
            return text

        # 한국어에서 한국어로 번역 요청하는 등 불필요한 번역 검출
        if target_language.lower() == 'korean' and (source_language.lower() == 'korean' or source_language.lower() == 'ko'):
            return text

        # 번역 실패 시 대체 메시지 설정
        if target_language == 'Korean':
            fallback_message = "죄송합니다. 서버 내 번역 기능에 일시적인 오류가 발생했습니다. 잠시 후 다시 시도해 주세요."
        else:
            fallback_message = text

        try:
            # 완벽한 번역 결과를 얻기 위해 시스템적인 포맷 지정
            prompt = (
                f"Translate the following text to {target_language}. "
                "Respond with ONLY the translated text, no other conversation or explanation:\n\n"
                f"---\n{text}\n---"
            )

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[prompt],
                config={"temperature": 0.0}  # 번역의 일관성을 위해 0.0 설정
            )

            translated_text = response.text.strip()
            # 마크다운 구분자 등 필요없는 문자열 제거
            if translated_text.startswith("---") and translated_text.endswith("---"):
                translated_text = translated_text.replace("---", "").strip()
            return translated_text
            
        except APIError as e:
            print(f"Gemini API 번역 중 오류 발생: {e}")
            return fallback_message
        except Exception as e:
            print(f"Gemini 번역 중 알 수 없는 예외 발생: {e}")
            return fallback_message
