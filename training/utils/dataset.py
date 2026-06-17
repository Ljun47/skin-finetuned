import json
import os
import random
import warnings
from typing import Dict, List, Any
import torch
from torch.utils.data import Dataset
from PIL import Image

# -----------------------------------
# 1차 SFT 학습용 Vision Dataset 클래스
# -----------------------------------
class VisionDataset(Dataset):
    def __init__(self, json_folder, json_file, image_folder, processor, max_length, data_limit=None, random_sample=True, random_seed=42):
        self.processor = processor
        self.max_length = max_length
        self.data = self._load_data_from_file(
            json_folder,
            json_file,
            image_folder,
            data_limit=data_limit,
            random_sample=random_sample,
            random_seed=random_seed
        )

        if len(self.data) == 0:
            raise ValueError(f"데이터가 없습니다! 파일 확인: {os.path.join(json_folder, json_file)}")

    def _load_data_from_file(self, json_folder, json_file, image_folder, data_limit=None, random_sample=True, random_seed=42):
        """
        JSONL 파일에서 데이터를 로드하고, 4줄씩 묶어 LLaVA 멀티턴 형식으로 변환 후 샘플링
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
                        continue

                    # 4줄씩 묶어 하나의 멀티턴 대화 세트로 처리
                    current_set.append(item)

                    if len(current_set) == 4:
                        # 이미지 경로 추출
                        image_full_path = current_set[0][DATA_IMAGE_KEY]
                        image_filename = os.path.basename(image_full_path)
                        image_path = os.path.join(image_folder, image_filename)

                        # LLaVA 멀티턴 프롬프트 구성
                        # 포맷: USER: {질문1} ASSISTANT: {답변1} USER: {질문2} ...
                        prompt_parts = []
                        for data_item in current_set:
                            q = data_item[DATA_QUESTION_KEY]
                            a = data_item[DATA_ANSWER_KEY]
                            prompt_parts.append(f"USER: {q} ASSISTANT: {a}")

                        prompt_text = " ".join(prompt_parts)

                        # 이미지 파일 존재 여부 확인 후 저장
                        if os.path.exists(image_path):
                            all_data_sets.append({
                                'image_path': image_path,
                                'prompt': prompt_text
                            })

                        current_set = []

                except json.JSONDecodeError as e:
                    print(f"❌ JSONL 파싱 실패 (Line {total_lines}): {e}")
                except Exception as e:
                    print(f"❌ 예상치 못한 오류 발생 (Line {total_lines}): {e}")

        total_sets = total_lines // 4
        print(f"총 JSONL 줄 수: {total_lines} (총 세트 수: {total_sets}개)")
        print(f"📊 이미지와 연결된 최종 유효 세트: {len(all_data_sets)}개")

        # 샘플 제한 및 셔플링
        target_limit = data_limit if data_limit is not None else len(all_data_sets)
        if target_limit > 0 and len(all_data_sets) > target_limit and random_sample:
            random.seed(random_seed)
            all_data_sets = random.sample(all_data_sets, target_limit)
            print(f"✨ 데이터 제한 설정 ({target_limit}개) 및 랜덤 샘플링 완료.")

        print(f"➡️ 최종 반환 데이터: {len(all_data_sets)}개")
        return all_data_sets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        try:
            image = Image.open(item['image_path']).convert('RGB')
            prompt = item['prompt']

            # LLaVA 포맷에 맞게 <image> 토큰을 첫 USER: 뒤에 삽입
            if prompt.startswith("USER: "):
                final_prompt = prompt.replace("USER: ", "USER: <image>\n", 1)
            else:
                final_prompt = "<image>\n" + prompt

            inputs = self.processor(
                image, final_prompt,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=self.max_length
            )

            # squeeze를 통해 배치 차원 제거
            for key in inputs:
                if inputs[key].dim() > 1:
                    inputs[key] = inputs[key].squeeze(0)

            # 라벨 생성 (첫 ASSISTANT: 이후 부분만 학습, 그 이전은 -100)
            labels = inputs['input_ids'].clone()
            assistant_tokens = self.processor.tokenizer.encode("ASSISTANT:", add_special_tokens=False)

            if assistant_tokens and assistant_tokens[0] in labels:
                assistant_token = assistant_tokens[0]
                try:
                    assistant_index = (labels == assistant_token).nonzero(as_tuple=True)[0][0]
                    labels[:assistant_index + 1] = -100
                except IndexError:
                    pass

            inputs['labels'] = labels
            return inputs

        except Exception as e:
            print(f"❌ 데이터 로드 실패 (인덱스 {idx}, 경로: {item.get('image_path', 'N/A')}): {e}")
            if idx == 0 and len(self.data) > 0:
                raise Exception("첫 번째 샘플을 로드할 수 없습니다.")
            return self.__getitem__(0)


# -----------------------------------
# 2차 RAFT 학습용 Dataset 클래스
# -----------------------------------
class DermaRaftTextDatasetWithDocs(Dataset):
    """
    RAFT 문맥 혼합 데이터셋
    __getitem__에서 매번 문맥 순서를 랜덤으로 섞고,
    그 중 어떤 번호가 golden인지 golden_doc_id로 반환
    """
    def __init__(self, path: str):
        self.samples: List[Dict[str, Any]] = []

        if not os.path.exists(path):
            raise FileNotFoundError(f"RAFT JSONL 파일 {path}이 존재하지 않습니다.")

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
            query = obj.get("query")
            golden = obj.get("golden", {}) or {}
            hard_negs = obj.get("hard_negatives", []) or []
            easy_neg = obj.get("easy_negative", None)

            golden_text = golden.get("text")
            golden_label = golden.get("label")

            if not query or not golden_text:
                continue

            self.samples.append({
                "query": query,
                "golden_text": golden_text,
                "golden_label": golden_label,
                "hard_negatives": hard_negs,
                "easy_negative": easy_neg,
            })

        print(f"✅ RAFT 텍스트+문맥 데이터 로드 완료: {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        base = self.samples[idx]

        query = base["query"]
        golden_text = base["golden_text"]
        golden_label = base["golden_label"]
        hard_negs = base["hard_negatives"]
        easy_neg = base["easy_negative"]

        # (role, text) 리스트 결합
        docs_raw = [("golden", golden_text)]
        for hn in hard_negs:
            docs_raw.append(("hard_negative", hn.get("text")))
        if easy_neg is not None:
            docs_raw.append(("easy_negative", easy_neg.get("text")))

        # 텍스트 없는 무효 문맥 필터링
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

        assert golden_doc_id is not None, "golden 문맥이 누락된 샘플이 있습니다."

        return {
            "question": query,
            "answer": golden_text,
            "label": golden_label,
            "docs": docs_with_id,
            "golden_doc_id": golden_doc_id,
        }


# -----------------------------------
# 2차 RAFT 학습용 Data Collator 클래스
# -----------------------------------
class DataCollatorRaftWithDocs:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        prompts: List[str] = []
        targets: List[str] = []

        for f in features:
            q = f["question"]
            answer_text = f["answer"]
            docs = f["docs"]
            golden_doc_id = f["golden_doc_id"]

            # 문맥 블록 문자열 구성 [1] ..., [2] ...
            docs_lines = []
            for d in docs:
                docs_lines.append(f"[{d['id']}] {d['text']}")
            docs_block = "\n".join(docs_lines)

            # LLaVA가 아닌 일반 텍스트 추론 또는 LLaVA prompt text 구성
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

        # 1) prompt만 인코딩 (길이 산출용)
        enc_prompt = self.processor(
            text=prompts,
            padding="longest",
            return_tensors="pt"
        )

        # 2) prompt + target 결합 인코딩
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

        # prompt 영역은 loss 연산에서 -100으로 제외
        for i in range(len(prompts)):
            prompt_ids = enc_prompt["input_ids"][i]
            prompt_len = (prompt_ids != pad_id).sum()
            labels[i, :prompt_len] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


# -----------------------------------
# SFT/LLaVA 학습용 collate_fn (Trainer에 단독 전달용)
# -----------------------------------
def sft_collate_fn(batch, processor):
    keys = batch[0].keys()
    batched = {}
    pad_token_id = processor.tokenizer.pad_token_id or 0

    for key in keys:
        tensors = [item[key] for item in batch]

        if key in ['input_ids', 'attention_mask', 'labels']:
            max_len = max(t.shape[0] for t in tensors)
            padded_tensors = []
            for tensor in tensors:
                if tensor.shape[0] < max_len:
                    pad_size = max_len - tensor.shape[0]
                    if key == 'labels':
                        pad_value = -100
                    elif key == 'attention_mask':
                        pad_value = 0
                    else:
                        pad_value = pad_token_id

                    padding = torch.full((pad_size,), pad_value, dtype=tensor.dtype)
                    tensor = torch.cat([tensor, padding])
                padded_tensors.append(tensor)
            batched[key] = torch.stack(padded_tensors)
        else:
            batched[key] = torch.stack(tensors)
    return batched
