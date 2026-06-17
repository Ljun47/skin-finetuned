import argparse
import os
import sys
import torch
from transformers import (
    AutoProcessor,
    AutoModelForVision2Seq,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model

# 패키지 경로 탐색을 위해 최상단 경로 추가
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from utils.env_helper import init_environment, clear_cuda_cache
from utils.dataset import VisionDataset, sft_collate_fn

class LlavaVisionTrainer:
    def __init__(self, args):
        self.args = args
        init_environment()
        clear_cuda_cache()
        
        print("📥 Processor 및 Tokenizer 로드 중...")
        self.processor = self._load_processor()
        
        print("📥 LLaVA Vision-Language 7B 모델 양자화 로드 중...")
        self.model = self._load_model()
        print("✅ 모델 및 어댑터 부착 완료!")

    def _load_processor(self):
        processor = AutoProcessor.from_pretrained(self.args.model_id)
        if processor.tokenizer.pad_token is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token
        return processor

    def _load_model(self):
        # 8비트 양자화 설정
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True
        )

        model = AutoModelForVision2Seq.from_pretrained(
            self.args.model_id,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16
        )

        # LLaVA 훈련을 위한 이미지 토큰 삽입 및 토크나이저 임베딩 레이어 조정
        if "<image>" not in self.processor.tokenizer.get_vocab():
            print("⚠️ <image> 토큰이 토크나이저에 없습니다. 강제 추가 후 임베딩을 조정합니다.")
            self.processor.tokenizer.add_tokens(["<image>"], special_tokens=True)
            model.resize_token_embeddings(len(self.processor.tokenizer))
            print("✅ <image> 토큰 임베딩 레이어 조정 완료.")

        # LoRA 적용 전 백본 파라미터 고정
        for param in model.parameters():
            param.requires_grad = False

        model.train()
        model.gradient_checkpointing_enable()

        # LoRA 설정 정의
        lora_config = LoraConfig(
            r=self.args.lora_r,
            lora_alpha=self.args.lora_alpha,
            target_modules=[
                "q_proj", "v_proj", "k_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ],
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, lora_config)

        if hasattr(model.config, 'pad_token_id') and model.config.pad_token_id is None:
            model.config.pad_token_id = self.processor.tokenizer.pad_token_id

        # 학습 가능 가중치 모니터링
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"🔍 학습 가능한 파라미터 수: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")
        model.print_trainable_parameters()
        
        return model

    def prepare_data(self):
        print("📊 SFT 학습 데이터셋 로드 및 LLaVA 멀티턴 세트 빌드 중...")
        
        train_limit = self.args.data_limit if self.args.use_limit else None
        
        self.train_dataset = VisionDataset(
            json_folder=self.args.data_folder,
            json_file=self.args.data_file,
            image_folder=self.args.image_folder,
            processor=self.processor,
            max_length=self.args.max_length,
            data_limit=train_limit,
            random_sample=True,
            random_seed=42
        )
        print(f"✅ 유효 훈련 턴 데이터셋 준비 완료: {len(self.train_dataset)}개 턴 세트")

    def run_training(self):
        print("🚀 Trainer 설정 및 SFT QLoRA 학습 개시...")
        
        # bf16 지원 여부 자동 검출
        use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8

        training_args = TrainingArguments(
            output_dir=self.args.output_dir,
            num_train_epochs=self.args.epochs,
            per_device_train_batch_size=self.args.batch_size,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=self.args.grad_accum,
            learning_rate=self.args.lr,
            weight_decay=0.01,
            max_grad_norm=1.0,
            warmup_steps=20,
            logging_steps=50,
            save_steps=100,
            save_total_limit=2,
            bf16=use_bf16,
            fp16=not use_bf16,
            gradient_checkpointing=True,
            remove_unused_columns=False,
            dataloader_pin_memory=False,
            dataloader_num_workers=0,
            report_to="none",
            ddp_find_unused_parameters=False
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            tokenizer=self.processor.tokenizer,
            data_collator=lambda b: sft_collate_fn(b, self.processor)
        )

        # 학습 구동 및 저장
        trainer.train()
        trainer.save_model()
        self.processor.save_pretrained(self.args.output_dir)
        print(f"🎉 SFT 학습 및 어댑터 저장 완료: {self.args.output_dir}")

def main():
    parser = argparse.ArgumentParser(description="LLaVA-1.5-7B 1차 SFT QLoRA 학습 CLI")
    
    # 모델 및 경로 아규먼트
    parser.add_argument("--model_id", type=str, default="llava-hf/llava-1.5-7b-hf", help="Hugging Face 베이스 모델 ID")
    parser.add_argument("--data_folder", type=str, default="/content/drive/MyDrive", help="데이터 파일이 위치한 폴더")
    parser.add_argument("--data_file", type=str, default="output_en.jsonl", help="학습용 JSONL 파일명")
    parser.add_argument("--image_folder", type=str, default="/content/drive/MyDrive/T_png", help="매칭되는 이미지 폴더")
    parser.add_argument("--output_dir", type=str, default="./finetuned_sft_output", help="학습된 LoRA 어댑터 저장 경로")
    
    # 학습 하이퍼파라미터
    parser.add_argument("--epochs", type=int, default=1, help="학습 에폭 수")
    parser.add_argument("--batch_size", type=int, default=16, help="디바이스별 학습 배치 크기")
    parser.add_argument("--grad_accum", type=int, default=2, help="그래디언트 누적 스텝 수")
    parser.add_argument("--lr", type=float, default=1e-4, help="러닝 레이트")
    parser.add_argument("--max_length", type=int, default=2048, help="최대 토큰 시퀀스 길이")
    
    # LoRA 설정
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA Rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA Alpha")
    
    # 데이터 제한 관련
    parser.add_argument("--use_limit", action="store_true", help="학습 데이터 제한 적용 여부")
    parser.add_argument("--data_limit", type=int, default=2500, help="최대 학습 데이터 턴 수")

    args = parser.parse_args()
    
    # 환경변수 로딩 후 경로 검증
    # 로컬 경로가 유효하지 않을 경우 탐색하여 보정
    for attr in ["data_folder", "image_folder"]:
        val = getattr(args, attr)
        if not os.path.exists(val):
            # 로컬 프로젝트 sample_data 폴더 등 대안 경로 설정
            alternatives = ["data/sample_data", "./data", "../../data"]
            for alt in alternatives:
                candidate = os.path.join(alt, os.path.basename(val)) if val.endswith("png") else alt
                if os.path.exists(candidate):
                    setattr(args, attr, candidate)
                    print(f"ℹ️ {attr} 경로 보정됨: {candidate}")
                    break
                    
    # SFT 파일 체크
    full_data_path = os.path.join(args.data_folder, args.data_file)
    if not os.path.exists(full_data_path):
        # /content/ 로딩에 맞춘 임시 로컬 덤프 생성 방지용 탐색
        for prefix in ["", "../../", "./", "data/sample_data/"]:
            candidate = os.path.join(prefix, args.data_file)
            if os.path.exists(candidate):
                args.data_folder = os.path.dirname(candidate)
                args.data_file = os.path.basename(candidate)
                print(f"ℹ️ 데이터 경로 보정됨: {candidate}")
                break

    trainer = LlavaVisionTrainer(args)
    trainer.prepare_data()
    trainer.run_training()

if __name__ == "__main__":
    main()
