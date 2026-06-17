import argparse
import os
import shutil
import sys
import torch
from transformers import (
    AutoProcessor,
    LlavaForConditionalGeneration,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from huggingface_hub import HfApi

# 패키지 경로 탐색을 위해 최상단 경로 추가
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from utils.env_helper import init_environment, clear_cuda_cache
from utils.dataset import DermaRaftTextDatasetWithDocs, DataCollatorRaftWithDocs

class LlavaRaftTrainer:
    def __init__(self, args):
        self.args = args
        init_environment()
        clear_cuda_cache()

        print(f"📥 1차 SFT 완료된 모델 및 Processor 로드 중: {args.sft_checkpoint}")
        self.processor = AutoProcessor.from_pretrained(args.sft_checkpoint)
        
        # 8비트 양자화 설정
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=3.0,
            llm_int8_has_fp16_weight=False
        )
        
        self.model = LlavaForConditionalGeneration.from_pretrained(
            args.sft_checkpoint,
            quantization_config=bnb_config,
            device_map="auto"
        )
        self.model.config.use_cache = False
        print("✅ 8bit 양자화 SFT 모델 로딩 완료.")

        # LoRA 어댑터 적용
        self.model = prepare_model_for_kbit_training(self.model)
        
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM"
        )
        self.model = get_peft_model(self.model, lora_config)
        print("🚀 LoRA 어댑터 설정 완료")
        self.model.print_trainable_parameters()

    def prepare_data(self):
        print(f"📊 RAFT 텍스트/문맥 데이터셋 빌드 중: {self.args.raft_data}")
        self.train_dataset = DermaRaftTextDatasetWithDocs(self.args.raft_data)
        self.data_collator = DataCollatorRaftWithDocs(self.processor)

    def run_training(self):
        print("🚀 Trainer 설정 및 RAFT QLoRA 학습 개시...")
        
        use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8

        training_args = TrainingArguments(
            output_dir=self.args.output_dir,
            num_train_epochs=self.args.epochs,
            per_device_train_batch_size=self.args.batch_size,
            gradient_accumulation_steps=self.args.grad_accum,
            learning_rate=self.args.lr,
            warmup_ratio=0.05,
            logging_steps=10,
            save_steps=100,
            save_total_limit=2,
            bf16=use_bf16,
            fp16=not use_bf16,
            gradient_checkpointing=True,
            remove_unused_columns=False,
            report_to="none"
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            data_collator=self.data_collator
        )

        trainer.train()
        trainer.save_model(self.args.output_dir)
        self.processor.save_pretrained(self.args.output_dir)
        print(f"✅ RAFT 학습 완료 및 LoRA 어댑터 저장: {self.args.output_dir}")

def merge_and_save_model(args):
    """
    베이스 모델과 학습 완료된 LoRA 어댑터를 병합하여 완전한 FP16 모델로 추출 저장합니다.
    """
    print("\n🔄 베이스 모델과 LoRA 어댑터 결합(Merge & Unload) 프로세스 시작...")
    
    # 1) Base 모델 로드
    base_model = LlavaForConditionalGeneration.from_pretrained(
        args.base_model_id,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="cpu" # 병합 시 OOM 방지를 위해 cpu 혹은 가벼운 단일 디바이스 할당
    )
    
    # 2) LoRA 어댑터 부착 및 결합
    model = PeftModel.from_pretrained(
        base_model,
        args.output_dir,
        torch_dtype=torch.float16
    )
    
    merged_model = model.merge_and_unload()
    
    # 3) 결과 저장
    merged_model.save_pretrained(args.merged_dir)
    processor = AutoProcessor.from_pretrained(args.base_model_id)
    processor.save_pretrained(args.merged_dir)
    print(f"✅ 모델 결합 완료 및 저장: {args.merged_dir}")

def upload_to_huggingface(args):
    """
    병합 완료된 모델을 지정한 Hugging Face 저장소에 푸시합니다.
    Inference Endpoint용 handler.py 및 README.md 템플릿도 자동으로 함께 업로드합니다.
    """
    print(f"\n📤 Hugging Face 저장소 업로드 시작: {args.repo_id}")
    
    hf_token = os.getenv("HF_TOKEN", "")
    if not hf_token:
        print("⚠️ [알림] 환경 변수 HF_TOKEN이 설정되지 않았습니다. 업로드를 진행할 수 없습니다.")
        return

    # 1) handler.py 템플릿 복사
    current_dir = os.path.dirname(os.path.abspath(__file__))
    src_handler = os.path.join(current_dir, "handler.py")
    dest_handler = os.path.join(args.merged_dir, "handler.py")
    
    if os.path.exists(src_handler):
        shutil.copy(src_handler, dest_handler)
        print("📝 Endpoint용 handler.py 파일 병합 폴더에 추가 완료.")
    else:
        print(f"⚠️ 경고: {src_handler} 템플릿 파일을 찾을 수 없어 handler.py 동적 추가를 생략합니다.")

    # 2) README.md 자동 생성
    readme_path = os.path.join(args.merged_dir, "README.md")
    if not os.path.exists(readme_path):
        readme_content = f"""---
license: llama2
base_model: {args.base_model_id}
tags:
- llava
- vision
- raft
- medical
- fine-tuned
language:
- ko
- en
pipeline_tag: image-text-to-text
---

# LLaVA 1.5 7B - RAFT 피부 질환 어라인먼트 모델

이 모델은 `{args.base_model_id}`를 기반으로 피부 질환 지식 데이터와 RAFT 기법을 사용하여 미세조정(Fine-tuned)한 멀티모달 모델입니다.
"""
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(readme_content)
        print("📝 기본 README.md 생성 완료.")

    # 3) Hugging Face Hub API 연동 및 업로드
    api = HfApi(token=hf_token)
    
    try:
        api.create_repo(
            repo_id=args.repo_id,
            private=True,
            exist_ok=True
        )
        print(f"✅ 저장소 존재 확인/생성 완료: {args.repo_id}")
        
        print("🚀 파일 일괄 업로드 업로드 진행 중 (대용량 파일 전송으로 시간이 소요됩니다)...")
        api.upload_folder(
            folder_path=args.merged_dir,
            repo_id=args.repo_id,
            commit_message="Upload fully merged LLaVA RAFT model with custom handler",
            ignore_patterns=[".git/*", "*.pyc", "__pycache__/*", ".DS_Store"]
        )
        print(f"🎉 Hugging Face 업로드 전과정 성공 완료! 저장소: https://huggingface.co/{args.repo_id}")
    except Exception as e:
        print(f"❌ 업로드 실패: {e}")

def main():
    parser = argparse.ArgumentParser(description="LLaVA-1.5-7B 2차 RAFT RAG 얼라인먼트 학습 및 배포 CLI")
    
    # 모델 및 데이터 경로
    parser.add_argument("--sft_checkpoint", type=str, default="/content/drive/MyDrive/dataset/finetuned_test_04", help="1차 SFT 가중치 체크포인트 폴더")
    parser.add_argument("--base_model_id", type=str, default="llava-hf/llava-1.5-7b-hf", help="Hugging Face 베이스 모델 ID")
    parser.add_argument("--raft_data", type=str, default="/content/raft_train_dataset_final.jsonl", help="RAFT 학습 데이터셋 JSONL 경로")
    
    # 아웃풋 저장 경로
    parser.add_argument("--output_dir", type=str, default="./llava_raft_adapter", help="2차 LoRA 어댑터 저장 경로")
    parser.add_argument("--merged_dir", type=str, default="./llava_raft_merged", help="최종 병합된 Full 가중치 저장 경로")
    
    # 하이퍼파라미터
    parser.add_argument("--epochs", type=int, default=1, help="학습 에폭 수")
    parser.add_argument("--batch_size", type=int, default=1, help="디바이스 배치 크기")
    parser.add_argument("--grad_accum", type=int, default=8, help="그래디언트 누적 스텝")
    parser.add_argument("--lr", type=float, default=6e-6, help="러닝 레이트")
    
    # LoRA 설정
    parser.add_argument("--lora_r", type=int, default=64, help="LoRA Rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA Alpha")
    
    # 허깅페이스 업로드 설정
    parser.add_argument("--repo_id", type=str, default="jayun/llava_raft_01", help="업로드할 HF Repository ID")
    parser.add_argument("--skip_train", action="store_true", help="학습 단계를 건너뛰고 기존 체크포인트 병합 및 업로드만 수행")

    args = parser.parse_args()

    # 입력 경로 정규화 및 보정
    if not os.path.exists(args.sft_checkpoint):
        # 로컬 세이브 폴더 등 탐색
        for alt in ["./finetuned_sft_output", "../stage1-sft/finetuned_sft_output"]:
            if os.path.exists(alt):
                args.sft_checkpoint = alt
                print(f"ℹ️ SFT 가중치 경로 보정됨: {alt}")
                break
                
    if not os.path.exists(args.raft_data):
        for prefix in ["", "../../", "./", "data/sample_data/"]:
            candidate = os.path.join(prefix, "raft_train_dataset_final.jsonl")
            if os.path.exists(candidate):
                args.raft_data = candidate
                print(f"ℹ️ RAFT 데이터 경로 보정됨: {candidate}")
                break

    # 1. 학습 실행
    if not args.skip_train:
        trainer = LlavaRaftTrainer(args)
        trainer.prepare_data()
        trainer.run_training()
    else:
        print("ℹ️ --skip_train 플래그가 설정되어 훈련을 건너뜁니다.")

    # 2. 모델 병합
    merge_and_save_model(args)
    
    # 3. 허깅페이스 업로드
    upload_to_huggingface(args)

if __name__ == "__main__":
    main()
