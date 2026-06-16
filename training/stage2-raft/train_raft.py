import os
from dotenv import load_dotenv

# W&B 로깅을 완전히 비활성화합니다.
os.environ['WANDB_MODE'] = 'disabled'

# --------------------------
# 💡 환경 변수(.env) 자동 스캔 및 로드
# --------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
root_env = os.path.join(current_dir, "../../.env")
backend_env = os.path.join(current_dir, "../../service/backend/.env")
local_env = os.path.join(current_dir, ".env")

if os.path.exists(local_env):
    load_dotenv(local_env)
elif os.path.exists(backend_env):
    load_dotenv(backend_env)
elif os.path.exists(root_env):
    load_dotenv(root_env)
else:
    load_dotenv()



# ==================================================
# [Code Cell]
# ==================================================
# # 깃허브 업로드 및 로컬 실행을 위해 pip 명령 주석 처리
# !pip install torch torchvision transformers
# !pip install accelerate peft
# !pip install pillow requests
# !pip install -U bitsandbytes



# ==================================================
# [Code Cell]
# ==================================================
# import torch
# import gc
# 
# # GPU 캐시 비우기
# torch.cuda.empty_cache()
# 
# # 가비지 컬렉션 실행
# gc.collect()
# 
# # VRAM 사용량 확인
# print(f"할당된 메모리: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
# print(f"예약된 메모리: {torch.cuda.memory_reserved()/1024**3:.2f} GB")



# ==================================================
# [Code Cell]
# ==================================================
# from google.colab import drive
# drive.mount('/content/drive')



"""
## **2차 파인튜닝 코드**
"""


# ==================================================
# [Code Cell]
# ==================================================
# # pip 설치 주석 처리
# !pip install -U "transformers>=4.40.0" accelerate safetensors datasets pillow

import os
import json
from dataclasses import dataclass
from typing import Dict, List, Any

import torch
from torch.utils.data import Dataset
from PIL import Image

from transformers import (
    AutoProcessor,
    LlavaForConditionalGeneration,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training



# ==================================================
# [Code Cell]
# ==================================================
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,              # ✅ 4bit → 8bit 양자화
    llm_int8_threshold=3.0,
    llm_int8_has_fp16_weight=False
)



# ==================================================
# [Code Cell]
# ==================================================
# 1차 SFT까지 끝난 체크포인트 경로 (너 환경에 맞게 수정)
SFT_CHECKPOINT_DIR = "/content/drive/MyDrive/dataset/finetuned_test_04"
BASE_MODEL_ID = "llava-hf/llava-1.5-7b-hf"

device = "cuda" if torch.cuda.is_available() else "cpu"

# processor는 SFT 체크포인트나 base 중 아무거나 써도 됨 (토크나이저 구조 동일하면 OK)
processor = AutoProcessor.from_pretrained(SFT_CHECKPOINT_DIR)

model = LlavaForConditionalGeneration.from_pretrained(
    SFT_CHECKPOINT_DIR,
    quantization_config=bnb_config,
    device_map="auto",
)

model.config.use_cache = False
print("✅ 8bit SFT 모델 로드 완료")

# 🔥🔥🔥 PEFT (LoRA) 설정 추가 🔥🔥🔥
# 1. k-bit 학습을 위해 모델을 준비 (8bit)
model = prepare_model_for_kbit_training(model)

# 2. LoRA 설정 정의 (R: 64, Alpha: 16은 흔히 사용되는 최적 설정)
lora_config = LoraConfig(
    r=64,
    lora_alpha=16,
    # LLaVA 1.5에서 Q, K, V, O (Attention Projection) 레이어에 LoRA 적용
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
)

# 3. 모델에 LoRA 어댑터 적용
model = get_peft_model(model, lora_config)
# 학습 가능한 파라미터 수를 확인하여 메모리 효율성 확인
model.print_trainable_parameters()
# 🔥🔥🔥 LoRA 설정 끝 🔥🔥🔥



# ==================================================
# [Code Cell]
# ==================================================
import json
import random
from typing import Dict, List, Any
from torch.utils.data import Dataset

class DermaRaftTextDatasetWithDocs(Dataset):
    """
    RAFT 텍스트 전용 데이터셋
    - query
    - golden (label, text)
    - hard_negatives: list of {label, text}
    - easy_negative: {label, text} or None

    __getitem__에서 매번 문맥 순서를 랜덤으로 섞고,
    그 중 어떤 번호가 golden인지 golden_doc_id로 반환.
    """

    def __init__(self, path: str):
        self.samples: List[Dict[str, Any]] = []

        # JSON 배열 vs JSONL 자동 감지
        with open(path, "r", encoding="utf-8") as f:
            first_char = f.read(1)
            f.seek(0)
            if first_char == "[":
                raw_list = json.load(f)
            else:
                raw_list = []
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    raw_list.append(json.loads(line))

        for obj in raw_list:
            query     = obj.get("query")
            golden    = obj.get("golden", {}) or {}
            hard_negs = obj.get("hard_negatives", []) or []
            easy_neg  = obj.get("easy_negative", None)

            golden_text = golden.get("text")
            golden_label = golden.get("label")

            if not query or not golden_text:
                # 필수 정보 없으면 스킵
                continue

            # raw 형태로 저장해두고, 실제 문맥 배열은 __getitem__에서 섞어서 생성
            self.samples.append({
                "query": query,
                "golden_text": golden_text,
                "golden_label": golden_label,
                "hard_negatives": hard_negs,
                "easy_negative": easy_neg,
            })

        print(f"✅ RAFT 텍스트+문맥 데이터 로드 완료: {len(self.samples)} samples")
        if len(self.samples) > 0:
            print("  예시 1개 (raw):", self.samples[0])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        base = self.samples[idx]

        query         = base["query"]
        golden_text   = base["golden_text"]
        golden_label  = base["golden_label"]
        hard_negs     = base["hard_negatives"]
        easy_neg      = base["easy_negative"]

        # (role, text) 리스트로 모으기
        docs_raw = [("golden", golden_text)]
        for hn in hard_negs:
            docs_raw.append(("hard_negative", hn.get("text")))
        if easy_neg is not None:
            docs_raw.append(("easy_negative", easy_neg.get("text")))

        # 텍스트 없는 것 제거
        docs_raw = [(role, txt) for role, txt in docs_raw if txt]

        # 매 샘플마다 문맥 순서를 랜덤 셔플
        random.shuffle(docs_raw)

        docs_with_id = []
        golden_doc_id = None
        for i, (role, txt) in enumerate(docs_raw, start=1):
            docs_with_id.append({
                "id": i,
                "role": role,
                "text": txt,
            })
            if role == "golden":
                golden_doc_id = i

        # golden_doc_id는 반드시 하나 있어야 함
        assert golden_doc_id is not None, "golden 문맥이 누락된 샘플이 있습니다."

        return {
            "question": query,
            "answer": golden_text,       # 실제 최종 질의 답변 텍스트
            "label": golden_label,
            "docs": docs_with_id,        # 섞인 문맥 리스트
            "golden_doc_id": golden_doc_id,
        }



# ==================================================
# [Code Cell]
# ==================================================
RAFT_TRAIN_JSONL_PATH = "/content/raft_train_dataset_final.jsonl"
train_dataset = DermaRaftTextDatasetWithDocs(RAFT_TRAIN_JSONL_PATH)



# ==================================================
# [Code Cell]
# ==================================================
from dataclasses import dataclass
from typing import Dict, List, Any
import torch

@dataclass
class DataCollatorRaftWithDocs:
    processor: Any

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        prompts: List[str] = []
        targets: List[str] = []

        for f in features:
            q = f["question"]
            answer_text = f["answer"]
            docs = f["docs"]
            golden_doc_id = f["golden_doc_id"]

            # 문맥 블록 문자열 구성
            # [1] ..., [2] ..., ...
            docs_lines = []
            for d in docs:
                docs_lines.append(f"[{d['id']}] {d['text']}")
            docs_block = "\n".join(docs_lines)

            # USER 프롬프트 구성 (한국어 설명 포함)
            prompt_text = (
                f"{self.processor.tokenizer.bos_token}"
                "USER:\n"
                f"질문: {q}\n\n"
                "아래는 이 질문과 관련 있을 수도 있고 없을 수도 있는 문맥들이다.\n"
                "각 문맥은 [번호]로 표시되어 있다.\n\n"
                f"{docs_block}\n\n"
                "위 문맥들 중에서 질문에 가장 적절한 문맥 번호 하나를 고르고,\n"
                "그 문맥을 근거로 답변하라.\n"
                "반드시 아래 형식을 따르라.\n\n"
                "형식:\n"
                "근거 문맥: [번호]\n"
                "답변: (질문에 대한 자세한 설명)\n\n"
                "ASSISTANT:"
            )

            # 정답 텍스트 (golden 문맥 번호 + golden_text)
            target_text = (
                f"근거 문맥: [{golden_doc_id}]\n"
                f"답변: {answer_text.strip()}"
            )

            prompts.append(prompt_text)
            targets.append(target_text)

        # 1) prompt만 인코딩
        enc_prompt = self.processor(
            text=prompts,
            padding="longest",
            return_tensors="pt"
        )

        # 2) prompt + target 같이 인코딩
        full_texts = [p + " " + t for p, t in zip(prompts, targets)]
        enc_full = self.processor(
            text=full_texts,
            padding="longest",
            return_tensors="pt"
        )

        input_ids = enc_full["input_ids"]
        attention_mask = enc_full["attention_mask"]

        labels = input_ids.clone()
        pad_id = self.processor.tokenizer.pad_token_id

        # prompt 부분은 loss에서 제외 (-100)
        for i in range(len(prompts)):
            prompt_ids = enc_prompt["input_ids"][i]
            prompt_len = (prompt_ids != pad_id).sum()
            labels[i, :prompt_len] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }



# ==================================================
# [Code Cell]
# ==================================================
data_collator = DataCollatorRaftWithDocs(processor=processor)



# ==================================================
# [Code Cell]
# ==================================================
RAFT_JSONL_PATH = "/content/raft_train_dataset_final.jsonl"
output_dir = "/content/llava_raft_stage4_raftdocs"

train_dataset = DermaRaftTextDatasetWithDocs(RAFT_JSONL_PATH)
data_collator = DataCollatorRaftWithDocs(processor)

use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8

training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=6e-6,
    warmup_ratio=0.05,
    logging_steps=10,
    save_steps=200,
    save_total_limit=2,
    bf16=use_bf16,
    fp16=not use_bf16,
    gradient_checkpointing=True,
    remove_unused_columns=False,
    report_to="none",
)



# ==================================================
# [Code Cell]
# ==================================================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
)

trainer.train()

trainer.save_model(output_dir)
processor.save_pretrained(output_dir)

print("✅ RAFT 8bit 2차 파인튜닝 완료:", output_dir)



# ==================================================
# [Code Cell]
# ==================================================
# 모델이 저장된 폴더가 /content/trained_model 이라고 가정
# Google Drive의 '내 드라이브' 안에 'Saved_Models'라는 폴더를 만들고 그곳에 복사
# 드라이브에 백업
# !cp -r "/content/llava_raft_stage4_raftdocs" "/content/drive/MyDrive/dataset/finetuned_test_RAFT_03"



"""
## **엔드포인트**
"""


# ==================================================
# [Code Cell]
# ==================================================
from transformers import AutoModelForVision2Seq, AutoProcessor
from peft import PeftModel

BASE_MODEL = "llava-hf/llava-1.5-7b-hf"          # 파인튜닝에 썼던 베이스
LORA_DIR   = "/content/llava_raft_stage4_raftdocs"  # LoRA 저장된 폴더
MERGED_DIR = "/content/llava_raft_merged"       # 합친 모델 저장할 위치

# 1) Base LLaVA 모델 로드
base = AutoModelForVision2Seq.from_pretrained(
    BASE_MODEL,
    torch_dtype="float16",
    low_cpu_mem_usage=True
)

# 2) LoRA 어댑터 로드 + 결합
model = PeftModel.from_pretrained(
    base,
    LORA_DIR,
    torch_dtype="float16"
)

# 3) LoRA를 base에 merge
model = model.merge_and_unload()

# 4) 합쳐진 모델 저장
model.save_pretrained(MERGED_DIR)

# 5) processor/tokenizer도 같이 복사
processor = AutoProcessor.from_pretrained(BASE_MODEL)
processor.save_pretrained(MERGED_DIR)

print("✅ Merge 완료! 저장 경로:", MERGED_DIR)



# ==================================================
# [Code Cell]
# ==================================================
from huggingface_hub import HfApi, upload_folder

HF_TOKEN = os.getenv("HF_TOKEN", "")
REPO_ID = "jayun/llava_raft_01"
MODEL_DIR = "/content/llava_raft_merged"

api = HfApi(token=HF_TOKEN)

api.create_repo(
    repo_id=REPO_ID,
    private=True,
    exist_ok=True
)

upload_folder(
    folder_path=MODEL_DIR,
    repo_id=REPO_ID,
    token=HF_TOKEN,
    commit_message="Upload merged LLaVA RAFT model"
)

print("✅ 업로드 완료:", f"https://huggingface.co/{REPO_ID}")



# ==================================================
# [Code Cell]
# ==================================================
import os

MERGED_DIR = "/content/llava_raft_merged"

# 저장된 파일 확인
print("📁 저장된 파일 목록:")
for root, dirs, files in os.walk(MERGED_DIR):
    level = root.replace(MERGED_DIR, '').count(os.sep)
    indent = ' ' * 2 * level
    print(f'{indent}{os.path.basename(root)}/')
    subindent = ' ' * 2 * (level + 1)
    for file in files:
        filepath = os.path.join(root, file)
        size_mb = os.path.getsize(filepath) / (1024*1024)
        print(f'{subindent}{file} ({size_mb:.2f} MB)')



# ==================================================
# [Code Cell]
# ==================================================
from huggingface_hub import HfApi
import os

HF_TOKEN = os.getenv("HF_TOKEN", "")
REPO_ID = "jayun/llava_raft_01"
MERGED_DIR = "/content/llava_raft_merged"

print("🚀 Hugging Face 업로드 시작...")
print(f"📁 업로드 경로: {MERGED_DIR}")
print(f"🎯 대상 Repo: {REPO_ID}")
print("-" * 60)

# README 생성 (없으면)
readme_path = os.path.join(MERGED_DIR, "README.md")
if not os.path.exists(readme_path):
    readme_content = """---
license: llama2
base_model: llava-hf/llava-1.5-7b-hf
tags:
- llava
- vision
- multimodal
- raft
- fine-tuned
language:
- en
pipeline_tag: image-text-to-text
---

# LLaVA 1.5 7B - RAFT Fine-tuned

이 모델은 `llava-hf/llava-1.5-7b-hf`를 RAFT (Retrieval Augmented Fine-Tuning) 기법으로 파인튜닝한 비전-언어 모델입니다.

## 모델 정보
- **Base Model**: llava-hf/llava-1.5-7b-hf
- **Model Size**: 7B parameters
- **Fine-tuning Method**: RAFT
- **Training Stages**: 2-stage fine-tuning

## 사용 방법
```python
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
import torch

# 모델 로드
model = AutoModelForVision2Seq.from_pretrained(
    "jayun/llava_raft_01",
    torch_dtype=torch.float16,
    device_map="auto"
)
processor = AutoProcessor.from_pretrained("jayun/llava_raft_01")

# 이미지 준비
image = Image.open("your_image.jpg")
prompt = "USER: <image>\\nDescribe this image in detail.\\nASSISTANT:"

# 추론
inputs = processor(text=prompt, images=image, return_tensors="pt").to("cuda", torch.float16)
output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
response = processor.decode(output[0], skip_special_tokens=True)
print(response)
```

## 모델 구조
- Vision Encoder: CLIP ViT-L/14
- Language Model: Vicuna-7B
- Total Parameters: ~7B
- Precision: FP16

## 학습 데이터
- Stage 1: 기본 파인튜닝
- Stage 2: RAFT documents를 활용한 추가 학습

## 제한사항
- 영어 위주로 학습됨
- 이미지 해상도: 336x336 권장
- GPU 메모리: 최소 16GB 필요
"""
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme_content)
    print("✅ README.md 생성 완료")

api = HfApi(token=HF_TOKEN)

# Repo 생성 (이미 있으면 무시)
try:
    api.create_repo(
        repo_id=REPO_ID,
        private=True,  # private으로 할지 public으로 할지 선택
        exist_ok=True
    )
    print(f"✅ Repo 확인/생성: {REPO_ID}")
except Exception as e:
    print(f"⚠️  Repo 생성 중 경고: {e}")

# 업로드 (대용량 파일이므로 시간이 걸립니다)
print("\n📤 파일 업로드 중... (13GB 정도라 10-30분 소요될 수 있습니다)")
print("   진행상황은 터미널에서 확인하세요.")

try:
    api.upload_folder(
        folder_path=MERGED_DIR,
        repo_id=REPO_ID,
        commit_message="Upload fully merged LLaVA RAFT model (13.5GB)",
        ignore_patterns=[".git/*", "*.pyc", "__pycache__/*", ".DS_Store"]
    )

    print("\n" + "=" * 60)
    print("✅ 업로드 완료!")
    print("=" * 60)
    print(f"🔗 모델 확인: https://huggingface.co/{REPO_ID}")
    print(f"🔗 파일 확인: https://huggingface.co/{REPO_ID}/tree/main")

except Exception as e:
    print(f"\n❌ 업로드 중 오류 발생:")
    print(f"   {str(e)}")
    print("\n💡 해결 방법:")
    print("   1. 인터넷 연결 확인")
    print("   2. HF 토큰 권한 확인 (write 권한 필요)")
    print("   3. Colab Pro 사용 시 더 안정적")



# ==================================================
# [Code Cell]
# ==================================================
# Colab에서 실행
MERGED_DIR = "/content/llava_raft_merged"

# handler.py 내용을 위 코드로 작성
handler_code = '''
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

            if isinstance(inputs.get("image"), str):
                image_data = base64.b64decode(inputs["image"])
                image = Image.open(BytesIO(image_data)).convert("RGB")
            else:
                image = inputs["image"]

            question = inputs.get("question", "Describe this image.")
            prompt = f"USER: <image>\\n{question}\\nASSISTANT:"

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
'''

# handler.py 저장
with open(f"{MERGED_DIR}/handler.py", "w") as f:
    f.write(handler_code)

print("✅ handler.py 생성 완료")

# Hugging Face에 업로드
from huggingface_hub import HfApi

HF_TOKEN = os.getenv("HF_TOKEN", "")
REPO_ID = "jayun/llava_raft_01"

api = HfApi(token=HF_TOKEN)

# handler.py만 업로드
api.upload_file(
    path_or_fileobj=f"{MERGED_DIR}/handler.py",
    path_in_repo="handler.py",
    repo_id=REPO_ID,
    commit_message="Add custom handler for Inference Endpoint"
)

print(f"✅ handler.py 업로드 완료!")
print(f"🔗 확인: https://huggingface.co/{REPO_ID}/blob/main/handler.py")


