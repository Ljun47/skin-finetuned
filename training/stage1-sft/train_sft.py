# ==================================================
# [Code Cell]
# ==================================================
import os

# W&B 로깅을 완전히 비활성화합니다.
os.environ['WANDB_MODE'] = 'disabled'



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
##**파인튜닝**
"""


# ==================================================
# [Code Cell]
# ==================================================
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoProcessor,
    AutoModelForVision2Seq,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model, TaskType
from PIL import Image
import json
import os
import glob
import random
import warnings

# 경고 무시 (선택적)
warnings.filterwarnings("ignore")

print("✅ 라이브러리 로드 완료")



# ==================================================
# [Code Cell]
# ==================================================
# 셀 2: 하이퍼파라미터 및 설정

# ====================================
# 📝 하이퍼파라미터 설정 (8비트, JSONL, 전체 학습 반영)
# ====================================

# 모델 설정
MODEL_CONFIG = {
    "model_id": "llava-hf/llava-1.5-7b-hf",
    "max_length": 2048
}

# QLoRA 설정 (작동하는 코드와 동일하게 수정)
# QLORA_CONFIG = {
#     "load_in_4bit": True,
#     "bnb_4bit_compute_dtype": torch.bfloat16,  # 수정: A100 최적화를 위해 float16 -> bfloat16
#     "bnb_4bit_quant_type": "nf4",
#     "bnb_4bit_use_double_quant": True
# }

# QLoRA / 양자화 설정 (이제 8비트)
QLORA_CONFIG = {
    "load_in_8bit": True,          # ✅ 4bit → 8bit
    # 선택 옵션들 (원하면 추가 가능)
    # "llm_int8_threshold": 6.0,
    # "llm_int8_has_fp16_weight": False,
}


# LoRA 설정 (작동하는 코드와 동일하게 수정)
LORA_CONFIG = {
    "r": 16,
    "lora_alpha": 32,
    "target_modules": [
        "q_proj", "v_proj", "k_proj", "o_proj",   # attention
        "gate_proj", "up_proj", "down_proj",      # MLP
        "embed_tokens", "lm_head"               # 수정: 학습 파라미터 수를 줄여 효율성 및 안정성 증가 (주석 처리)
    ],
    "lora_dropout": 0.1,
    "bias": "none",
    "task_type": "CAUSAL_LM"
}

# 훈련 설정
# TRAINING_CONFIG = {
#     "output_dir": "./finetuned_test",
#     "num_train_epochs": 1,
#     "per_device_train_batch_size": 32,   # A100 기반 최대 효율 값 (EBS=128)
#     "per_device_eval_batch_size": 1,
#     "gradient_accumulation_steps": 4,     # 최소 오버헤드
#     "learning_rate": 2e-4,                # 수정: 학습률을 8e-4에서 6e-4로 보수적으로 낮춰 안정성 강화
#     "weight_decay": 0.01,
#     "max_grad_norm": 1.0,
#     "warmup_steps": 50,
#     "logging_steps": 10,
#     "save_steps": 300,                    # 에폭당 1회 저장 (총 900 스텝)
#     "eval_steps": 300                     # 에폭당 1회 평가
# }

TRAINING_CONFIG = {
    "output_dir": "./finetuned_test_01",

    "num_train_epochs": 1,
    "per_device_train_batch_size": 16,
    "per_device_eval_batch_size": 1,
    "gradient_accumulation_steps": 2,
    "learning_rate": 1e-4,
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    "warmup_steps": 20,
    "logging_steps": 50,
    "save_steps": 50,
    "eval_steps": 50,
}


# 📁 데이터 폴더 설정 (여기서 수정!)
DATA_CONFIG = {
        # JSONL 파일이 있는 폴더 (여기서는 /content/drive/MyDrive/dataset)
    "train_data_folder": "/content/drive/MyDrive",
    # JSONL 파일명
    "train_data_file": "output_en.jsonl",
    "train_image_folder": "/content/drive/MyDrive/T_png",
    "val_jsonl_file": None,
    "val_image_folder": None
}

# 🔢 데이터 양 조절 설정 (여기서 수정!)
DATA_LIMIT_CONFIG = {
    "use_data_limit": False,
    "train_data_limit": 2500,
    "val_data_limit": 1000,
    "random_sample": True,
    "random_seed": 42
}

# 생성 설정
GENERATION_CONFIG = {
    "max_new_tokens": 200,
    "temperature": 0.7,
    "do_sample": True
}


print("✅ 설정 로드 완료")



# ==================================================
# [Code Cell]
# ==================================================
# 셀 3: JSONL 데이터 로더 및 데이터셋 클래스

# ====================================
# 📚 JSONL 데이터 로더 함수
# ====================================

# 셀 3 - 수정된 JSONL 데이터 로더 함수 (멀티턴 'text' 키 처리)

def load_data_from_file(json_folder, json_file, image_folder, data_limit=None, random_sample=True, random_seed=42):
    """
    JSONL 파일에서 데이터를 로드하고, 4줄씩 묶어 LLaVA 멀티턴 형식으로 변환 후,
    data_limit만큼 랜덤 샘플링합니다.
    """
    all_data_sets = []
    file_path = os.path.join(json_folder, json_file)

    if not os.path.exists(file_path):
        print(f"⚠️ 경고: {file_path} 파일이 존재하지 않습니다.")
        return all_data_sets

    print(f"📄 JSONL 파일 로드 중: {file_path}")

    total_lines = 0
    missing_key_count = 0
    current_set = []

    # 수정: 실제 데이터 키를 정의합니다.
    DATA_IMAGE_KEY = 'image_path'
    DATA_QUESTION_KEY = 'question'
    DATA_ANSWER_KEY = 'answer'

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            total_lines += 1
            try:
                item = json.loads(line.strip())

                # 필수 키 확인
                if DATA_IMAGE_KEY not in item or DATA_QUESTION_KEY not in item or DATA_ANSWER_KEY not in item:
                    missing_key_count += 1
                    if missing_key_count <= 5:
                        print(f"❌ 필수 키 누락 (Line {total_lines}): 현재 키 {list(item.keys())}")
                    continue

                # ✅ 4줄씩 묶어서 하나의 세트로 처리
                current_set.append(item)

                # 4줄이 모이면 하나의 세트로 처리합니다.
                if len(current_set) == 4:

                    # 1. 이미지 경로 추출 (세트 내 모든 항목의 경로가 같다고 가정)
                    image_full_path = current_set[0][DATA_IMAGE_KEY]
                    image_filename = os.path.basename(image_full_path)
                    image_path = os.path.join(image_folder, image_filename)

                    # 2. LLaVA 멀티턴 프롬프트 구성
                    # LLaVA 형식: USER: <image>\n{질문1} ASSISTANT: {답변1} USER: {질문2} ASSISTANT: {답변2} ...
                    # <image> 태그는 VisionDataset.__getitem__에서 삽입됩니다.

                    prompt_parts = []
                    for data_item in current_set:
                        q = data_item[DATA_QUESTION_KEY]
                        a = data_item[DATA_ANSWER_KEY]
                        # NOTE: 첫 턴에만 \n을 넣고, 이후 턴은 줄바꿈 없이 붙입니다.
                        # VisionDataset에서 <image> 태그가 첫 USER: 뒤에 삽입됩니다.
                        prompt_parts.append(f"USER: {q} ASSISTANT: {a}")

                    # 모든 턴을 결합합니다.
                    prompt_text = " ".join(prompt_parts)

                    # 3. 이미지 파일 존재 여부 확인 후 저장
                    if os.path.exists(image_path):
                        all_data_sets.append({
                            'image_path': image_path,
                            'prompt': prompt_text
                        })
                    # else: 이미지 없음 처리 (기존 로직과 동일하게 생략)

                    # 세트 초기화
                    current_set = []

            except json.JSONDecodeError as e:
                print(f"❌ JSONL 파싱 실패 (Line {total_lines}): {e}")
            except Exception as e:
                print(f"❌ 예상치 못한 오류 발생 (Line {total_lines}): {e}")

    # --- 로딩 통계 및 샘플링 ---
    total_sets = total_lines // 4 # 총 줄 / 4 = 총 세트 수
    print(f"\n--- 로딩 통계 ---")
    print(f"총 JSONL 줄 수: {total_lines} (총 세트 수: {total_sets}개)")
    print(f"필수 키 누락 샘플 수: {missing_key_count}")
    print(f"📊 이미지와 연결된 최종 유효 세트: {len(all_data_sets)}개")

    # ✅ 2500개 랜덤 샘플링 로직
    # data_limit을 2500으로 설정하고 random_sample을 True로 설정해야 합니다.
    target_limit = data_limit if data_limit is not None else len(all_data_sets)

    if target_limit > 0 and len(all_data_sets) > target_limit and random_sample:
        random.seed(random_seed)
        all_data_sets = random.sample(all_data_sets, target_limit)
        print(f"✨ 데이터 제한 설정 ({target_limit}개) 및 랜덤 샘플링 완료.")

    # 💡 data_limit이 2500일 때, 최종 반환 데이터는 2500개가 됩니다.
    print(f"➡️ 최종 반환 데이터: {len(all_data_sets)}개")

    return all_data_sets

# ====================================
# 📚 데이터셋 클래스 (JSONL 로더 사용)
# ====================================

class VisionDataset(Dataset):
    def __init__(self, json_folder, json_file, image_folder, processor, max_length, data_limit=None, random_sample=True, random_seed=42):
        self.data = load_data_from_file(
            json_folder,
            json_file,
            image_folder,
            data_limit=data_limit,
            random_sample=random_sample,
            random_seed=random_seed
        )
        self.processor = processor
        self.max_length = max_length

        if len(self.data) == 0:
            raise ValueError(f"데이터가 없습니다! 파일 확인: {os.path.join(json_folder, json_file)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        try:
            # 이미지 로드
            image = Image.open(item['image_path']).convert('RGB')

            # 프롬프트 구성
            prompt = item['prompt'] # "USER: {질문}\nASSISTANT: {답변}"

            # ✅ 수정: LLaVA 형식에 맞게 <image> 토큰을 USER: 뒤에 삽입
            # prompt는 "USER: ..." 로 시작하므로, 첫 번째 "USER: "를 "USER: <image>\n"로 대체
            if prompt.startswith("USER: "):
                final_prompt = prompt.replace("USER: ", "USER: <image>\n", 1)
            else:
                final_prompt = "<image>\n" + prompt

            # 인코딩
            inputs = self.processor(
                image, final_prompt, # <--- final_prompt 사용
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=self.max_length
            )

            # 배치 차원 제거 (collate_fn에서 다시 추가)
            for key in inputs:
                if inputs[key].dim() > 1:
                    inputs[key] = inputs[key].squeeze(0)

            # 라벨 생성 (첫 ASSISTANT: 이후 부분만 학습, -100 사용)
            labels = inputs['input_ids'].clone()

            # 토크나이저를 통해 'ASSISTANT:' 토큰 ID 찾기
            assistant_tokens = self.processor.tokenizer.encode("ASSISTANT:", add_special_tokens=False)

            if assistant_tokens and assistant_tokens[0] in labels:
                assistant_token = assistant_tokens[0]

                # 첫 번째 'ASSISTANT:' 토큰 ID가 처음 나타나는 위치 찾기
                try:
                    assistant_index = (labels == assistant_token).nonzero(as_tuple=True)[0][0]
                    # 그 위치까지 (ASSISTANT: 토큰 자체 포함) -100으로 마스킹
                    labels[:assistant_index + 1] = -100
                except IndexError:
                    pass

            inputs['labels'] = labels
            return inputs

        except Exception as e:
            print(f"❌ 데이터 로드 실패 (인덱스 {idx}, 경로: {item.get('image_path', 'N/A')}): {e}")
            if idx == 0 and len(self.data) > 0:
                 raise Exception("첫 번째 샘플도 로드할 수 없습니다.")
            return self.__getitem__(0)

def collate_fn(self, batch):
    """배치 처리 - 크기 다른 텐서들을 패딩으로 맞춤"""
    # 유효한 샘플만 필터링 (VisionDataset에서 오류 발생 시 None 대신 다른 샘플 반환하도록 구현했으므로,
    # 여기서는 오류난 샘플을 고려하지 않습니다. 만약 VisionDataset이 None을 반환하면 필터링 로직이 필요합니다.)
    # batch = [item for item in batch if item is not None]

    keys = batch[0].keys()
    batched = {}

    # 패딩 토큰 ID 설정
    pad_token_id = self.processor.tokenizer.pad_token_id or 0

    for key in keys:
        tensors = [item[key] for item in batch]

        if key in ['input_ids', 'attention_mask', 'labels']:
            # 가장 긴 시퀀스 길이 찾기
            max_len = max(t.shape[0] for t in tensors)

            padded_tensors = []
            for tensor in tensors:
                if tensor.shape[0] < max_len:
                    pad_size = max_len - tensor.shape[0]

                    # 패딩 값 설정
                    if key == 'labels':
                        pad_value = -100
                    elif key == 'attention_mask':
                        pad_value = 0
                    else:  # input_ids
                        pad_value = pad_token_id

                    # 패딩 추가 (왼쪽 패딩을 고려해야 하지만, LLaVA는 오른쪽 패딩이 일반적)
                    # VisionDataset에서 이미 오른쪽 패딩으로 input_ids를 생성했으므로, 여기서도 오른쪽 패딩을 적용합니다.
                    padding = torch.full((pad_size,), pad_value, dtype=tensor.dtype)
                    tensor = torch.cat([tensor, padding])

                padded_tensors.append(tensor)

            batched[key] = torch.stack(padded_tensors)
        else:
            # pixel_values 등은 그대로
            batched[key] = torch.stack(tensors)

    return batched

print("✅ 데이터 로더 및 데이터셋 클래스 정의 완료")



# ==================================================
# [Code Cell]
# ==================================================
# ====================================
# 🚀 메인 트레이너 클래스 (수정됨)
# ====================================

class SimpleVisionTrainer:
    def __init__(self):
        print("📥 모델 로딩 중...")
        self.processor = self._load_processor()
        self.model = self._load_model()
        print("✅ 모델 로드 완료!")

    def _load_processor(self):
        processor = AutoProcessor.from_pretrained(MODEL_CONFIG["model_id"])
        if processor.tokenizer.pad_token is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token
        return processor

    def _load_model(self):
        # QLoRA 설정 (8비트 로드)
        bnb_config = BitsAndBytesConfig(**QLORA_CONFIG)

        # 모델 로드 (8비트 로드)
        model = AutoModelForVision2Seq.from_pretrained(
            MODEL_CONFIG["model_id"],
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16
        )

        # 🔧 수정: LLaVA 모델 훈련을 위해 <image> 토큰 추가 및 임베딩 크기 조정
        if "<image>" not in self.processor.tokenizer.get_vocab():
            print("⚠️ <image> 토큰이 토크나이저에 없습니다. 추가 후 모델 임베딩 크기를 조정합니다.")
            self.processor.tokenizer.add_tokens(["<image>"], special_tokens=True)

            # 새 토큰에 맞게 모델의 임베딩 레이어 크기를 조정합니다. (PEFT 적용 전)
            # 이 작업이 'Image features and image tokens do not match' 오류를 해결합니다.
            model.resize_token_embeddings(len(self.processor.tokenizer))
            print("✅ <image> 토큰 추가 및 모델 임베딩 크기 조정 완료.")

        # 🔧 LoRA 적용 전에 모든 파라미터 freeze
        for param in model.parameters():
            param.requires_grad = False

        model.train()
        model.gradient_checkpointing_enable()

        # LoRA 적용
        lora_config = LoraConfig(**LORA_CONFIG)
        model = get_peft_model(model, lora_config)

        # 패딩 토큰 설정
        if hasattr(model.config, 'pad_token_id') and model.config.pad_token_id is None:
            model.config.pad_token_id = self.processor.tokenizer.pad_token_id

        # 🔧 학습 가능한 파라미터 확인
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"🔍 학습 가능한 파라미터: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")

        model.print_trainable_parameters()
        return model

    def prepare_data(self):
        # ... (이전 코드와 동일, 생략)
        print("📊 데이터 준비 중...")

        # 훈련 데이터 (JSONL 파일명 사용)
        train_limit = DATA_LIMIT_CONFIG["train_data_limit"] if DATA_LIMIT_CONFIG["use_data_limit"] else None

        self.train_dataset = VisionDataset(
            DATA_CONFIG["train_data_folder"],
            DATA_CONFIG["train_data_file"], # JSONL 파일명 전달
            DATA_CONFIG["train_image_folder"],
            self.processor,
            MODEL_CONFIG["max_length"],
            data_limit=train_limit,
            random_sample=DATA_LIMIT_CONFIG["random_sample"],
            random_seed=DATA_LIMIT_CONFIG["random_seed"]
        )

        # 검증 데이터 (생략)
        self.val_dataset = None

        print(f"✅ 훈련 데이터: {len(self.train_dataset)}개")
        if self.val_dataset:
            print(f"✅ 검증 데이터: {len(self.val_dataset)}개")

    def collate_fn(self, batch):
        """배치 처리 - 크기 다른 텐서들을 패딩으로 맞춤"""
        keys = batch[0].keys()
        batched = {}

        # 패딩 토큰 ID 설정
        pad_token_id = self.processor.tokenizer.pad_token_id or 0

        for key in keys:
            tensors = [item[key] for item in batch]

            if key in ['input_ids', 'attention_mask', 'labels']:
                max_len = max(t.shape[0] for t in tensors)
                padded_tensors = []
                for tensor in tensors:
                    if tensor.shape[0] < max_len:
                        pad_size = max_len - tensor.shape[0]

                        # 패딩 값 설정
                        if key == 'labels':
                            pad_value = -100
                        elif key == 'attention_mask':
                            pad_value = 0
                        else:  # input_ids
                            pad_value = pad_token_id

                        # 패딩 추가
                        padding = torch.full((pad_size,), pad_value, dtype=tensor.dtype)
                        tensor = torch.cat([tensor, padding])

                    padded_tensors.append(tensor)

                batched[key] = torch.stack(padded_tensors)
            else:
                # pixel_values 등은 그대로
                batched[key] = torch.stack(tensors)

        return batched

    def train(self):
        print("🚀 훈련 시작!")

        # 훈련 인자
        training_args = TrainingArguments(
            eval_strategy="steps" if self.val_dataset else "no",
            save_strategy="steps",
            load_best_model_at_end=True if self.val_dataset else False,
            remove_unused_columns=False,
            dataloader_pin_memory=False,
            bf16=torch.cuda.is_bf16_supported(),
            fp16=not torch.cuda.is_bf16_supported(),
            gradient_checkpointing=True,
            dataloader_num_workers=0,
            report_to=None,
            ddp_find_unused_parameters=False,
            **TRAINING_CONFIG
        )

        # 트레이너
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            tokenizer=self.processor.tokenizer,
            data_collator=self.collate_fn, # collate_fn을 Trainer에 전달
        )

        # 훈련 실행
        trainer.train()

        # 저장
        trainer.save_model()
        self.processor.save_pretrained(TRAINING_CONFIG["output_dir"])

        print(f"✅ 완료! 모델 저장: {TRAINING_CONFIG['output_dir']}")
        return trainer

    def test(self, image_path, question):
        # ... (이전 코드와 동일, 생략)
        """테스트 생성"""
        image = Image.open(image_path).convert('RGB')
        # LLaVA 프롬프트 형식: USER: <image>\n{question} ASSISTANT:
        prompt = f"USER: <image>\n{question} ASSISTANT:"

        inputs = self.processor(image, prompt, return_tensors='pt').to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                **GENERATION_CONFIG
            )

        response = self.processor.decode(outputs[0], skip_special_tokens=True)
        answer = response.split("ASSISTANT:")[-1].strip() if "ASSISTANT:" in response else response

        return answer

print("✅ 트레이너 클래스 정의 완료")



# ==================================================
# [Code Cell]
# ==================================================
# 셀 5: 훈련 실행

def main():
    print("🔢 데이터 제한 설정:")
    if DATA_LIMIT_CONFIG["use_data_limit"]:
        print(f"  📊 훈련 데이터 제한: {DATA_LIMIT_CONFIG['train_data_limit']}개")
        print(f"  🎲 랜덤 샘플링: {DATA_LIMIT_CONFIG['random_sample']}")
    else:
        print("  📊 전체 데이터 사용 (제한 없음)")
    print()

    # 폴더/파일 확인
    print("📁 데이터 폴더/파일 확인...")
    print(f"✅ 훈련 JSONL 파일: {os.path.join(DATA_CONFIG['train_data_folder'], DATA_CONFIG['train_data_file'])}")
    print(f"✅ 훈련 이미지 폴더: {DATA_CONFIG['train_image_folder']}")
    if not os.path.exists(os.path.join(DATA_CONFIG['train_data_folder'], DATA_CONFIG['train_data_file'])):
         print(f"❌ 오류: 훈련 JSONL 파일이 없습니다! 경로를 확인하세요.")
         return
    if not os.path.exists(DATA_CONFIG['train_image_folder']):
         print(f"❌ 오류: 훈련 이미지 폴더가 없습니다! 경로를 확인하세요.")
         return

    # 트레이너 초기화 (모델 로드 및 LoRA 적용)
    trainer = SimpleVisionTrainer()

    # 데이터 준비 (JSONL 로드 및 데이터셋 생성)
    trainer.prepare_data()

    # 훈련
    trainer.train()

    print("🎉 훈련 완료!")

if __name__ == "__main__":
    # 로컬 실행을 위해 Colab 드라이브 마운트 주석 처리
    # from google.colab import drive
    # if not os.path.exists("/content/drive"):
    #      drive.mount("/content/drive")
    # else:
    #      print("Google Drive가 이미 마운트되어 있습니다.")

    main()



# ==================================================
# [Code Cell]
# ==================================================
# # 깃허브용 컴파일 에러 방지 위해 백업 명령어 주석 처리
# # !cp -r /content/finetuned_test_01 '/content/drive/MyDrive/dataset/finetuned_test_04'


