from typing import Dict, List, Any
from PIL import Image
import torch
import base64
from io import BytesIO

class EndpointHandler:
    def __init__(self, path=""):
        from transformers import AutoModelForVision2Seq, AutoProcessor

        self.model = AutoModelForVision2Seq.from_pretrained(
            path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(path)
        self.model.eval()

    def __call__(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        try:
            inputs = data.get("inputs", {})

            # Base64 이미지 디코딩 처리
            if isinstance(inputs.get("image"), str):
                image_data = base64.b64decode(inputs["image"])
                image = Image.open(BytesIO(image_data)).convert("RGB")
            else:
                image = inputs["image"]

            question = inputs.get("question", "Describe this image.")
            
            # LLaVA 프롬프트 템플릿 조립
            prompt = f"USER: <image>\n{question}\nASSISTANT:"

            model_inputs = self.processor(
                text=prompt,
                images=image,
                return_tensors="pt"
            ).to(self.model.device)

            with torch.no_grad():
                output = self.model.generate(
                    **model_inputs,
                    max_new_tokens=200,
                    do_sample=False
                )

            response = self.processor.decode(output[0], skip_special_tokens=True)

            if "ASSISTANT:" in response:
                response = response.split("ASSISTANT:")[-1].strip()

            return [{"generated_text": response}]

        except Exception as e:
            return [{"error": str(e)}]
