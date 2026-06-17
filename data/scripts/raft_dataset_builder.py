import json
import os
import random
from typing import Dict, List, Any
import numpy as np

class RaftDatasetBuilder:
    def __init__(self, hard_negative_mapping=None):
        # 도메인 매핑 정의 (필요시 클래스 외부에서 주입 가능)
        self.hard_negative_mapping = hard_negative_mapping or {
            "Atopic Dermatitis": "Psoriasis",
            "Psoriasis": "Atopic Dermatitis",
            "Seborrheic Dermatitis": "Rosacea",
            "Rosacea": "Seborrheic Dermatitis",
            "Acne": "Seborrheic Dermatitis",
            "Normal": "Acne"
        }

    def load_questions_data(self, questions_file_path: str) -> List[Dict[str, Any]]:
        """
        텍스트 파일로부터 질문과 골든 답변 데이터 셋을 파싱하여 로드
        """
        if not os.path.exists(questions_file_path):
            raise FileNotFoundError(f"질문 데이터 파일 {questions_file_path}이 존재하지 않습니다.")

        print(f"질문 데이터 로드 중: {questions_file_path}")
        questions_data = []
        
        with open(questions_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # 다양한 포맷 대응용 파싱 로직
        # 1. '### 번호' 형태
        if "###" in content:
            sections = content.split("###")
            for section in sections:
                if not section.strip():
                    continue
                lines = [line.strip() for line in section.strip().split('\n') if line.strip()]
                if len(lines) >= 3:
                    try:
                        # 첫 줄에서 번호와 레이블 파싱
                        first_line = lines[0]
                        # 예: "1. Rosacea (주사비)" -> "Rosacea" 추출
                        label = "Unknown"
                        for key in self.hard_negative_mapping.keys():
                            if key.lower() in first_line.lower():
                                label = key
                                break
                        
                        question = ""
                        answer = ""
                        q_idx, a_idx = -1, -1
                        for idx, line in enumerate(lines):
                            if line.startswith("질문:") or line.startswith("Q:"):
                                q_idx = idx
                            elif line.startswith("답변:") or line.startswith("A:"):
                                a_idx = idx
                        
                        if q_idx != -1 and a_idx != -1:
                            question = lines[q_idx].split(":", 1)[1].strip()
                            # 답변은 여러 줄일 수 있으므로 취합
                            answer_parts = []
                            for idx in range(a_idx, len(lines)):
                                line_content = lines[idx]
                                if idx == a_idx:
                                    answer_parts.append(line_content.split(":", 1)[1].strip())
                                else:
                                    if line_content.startswith("질문:") or line_content.startswith("Q:") or line_content.startswith("###"):
                                        break
                                    answer_parts.append(line_content)
                            answer = "\n".join(answer_parts)
                            
                            questions_data.append({
                                "label": label,
                                "query": question,
                                "answer": answer
                            })
                    except Exception as e:
                        print(f"세션 파싱 실패 경고: {e}")
        # 2. JSONL 형태인지 체크
        else:
            try:
                lines = content.strip().split('\n')
                for line in lines:
                    if line.strip():
                        questions_data.append(json.loads(line))
            except Exception as e:
                print(f"대체 파싱 실패: {e}. 일반 텍스트 포맷으로 간주합니다.")
                
        print(f"성공적으로 {len(questions_data)}개의 질문을 로드했습니다.")
        return questions_data

    def load_metadata(self, metadata_file_path: str) -> List[Dict[str, Any]]:
        """
        정제 완료된 지식 베이스(JSONL) 파일을 로드
        """
        if not os.path.exists(metadata_file_path):
            raise FileNotFoundError(f"지식 베이스 파일 {metadata_file_path}이 존재하지 않습니다.")

        documents = []
        with open(metadata_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        documents.append(json.loads(line))
                    except Exception as e:
                        print(f"지식베이스 파싱 에러: {e}")
        print(f"지식 베이스 로드 완료: {len(documents)}개 문서")
        return documents

    def calculate_similarity(self, query_text: str, candidate_texts: List[str]) -> np.ndarray:
        """
        TF-IDF 기반 코사인 유사도를 직접 계산
        (외부 scikit-learn 의존성을 최소화하기 위한 순수 파이썬/numpy 구현)
        """
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        
        if not candidate_texts:
            return np.zeros(0)
            
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([query_text] + candidate_texts)
        
        # 첫 번째 벡터가 query, 나머지가 candidates
        query_vector = tfidf_matrix[0:1]
        candidate_vectors = tfidf_matrix[1:]
        
        sims = cosine_similarity(query_vector, candidate_vectors)
        return sims[0]

    def get_hard_negatives(self, query: str, oracle_samples: List[Dict[str, Any]], count: int = 4) -> List[Dict[str, Any]]:
        """
        유사도 계산을 통해 질문과 관련성이 높은 Hard Negative 문맥을 선별
        """
        if not oracle_samples:
            return []
            
        candidate_texts = [doc.get('text', '') for doc in oracle_samples]
        similarities = self.calculate_similarity(query, candidate_texts)
        
        # 유사도 순으로 정렬된 인덱스
        sorted_indices = np.argsort(similarities)[::-1]
        
        hard_negs = []
        for idx in sorted_indices:
            doc = oracle_samples[idx]
            # 이미 선택된 문서 중복 방지
            if doc not in hard_negs:
                hard_negs.append({
                    "label": doc.get("label", ""),
                    "text": doc.get("text", ""),
                    "source": doc.get("source", ""),
                    "title": doc.get("title", "")
                })
                if len(hard_negs) >= count:
                    break
        return hard_negs

    def get_easy_negative(self, distractor_samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Distractor 그룹에서 무작위로 하나의 Easy Negative 문맥을 선별
        """
        if not distractor_samples:
            return {}
        doc = random.choice(distractor_samples)
        return {
            "label": doc.get("label", ""),
            "text": doc.get("text", ""),
            "source": doc.get("source", ""),
            "title": doc.get("title", "")
        }

    def generate_raft_samples(self, questions_file: str, candidate_file: str, output_file: str, hard_neg_count: int = 4):
        """
        골든 샘플 질문 데이터와 지식 데이터를 결합하여 RAFT 포맷의 데이터셋을 생성
        """
        questions = self.load_questions_data(questions_file)
        kb_docs = self.load_metadata(candidate_file)
        
        # Oracle과 Distractor 분리
        oracle_docs = [doc for doc in kb_docs if doc.get('document_type') == 'oracle']
        distractor_docs = [doc for doc in kb_docs if doc.get('document_type') == 'distractor']
        
        print(f"Oracle 분류 문서: {len(oracle_docs)}개, Distractor 분류 문서: {len(distractor_docs)}개")
        
        raft_dataset = []
        success_count = 0
        
        for q_idx, q in enumerate(questions):
            q_label = q.get('label')
            query = q.get('query')
            answer = q.get('answer')
            
            if not query or not answer:
                continue
                
            # Golden Document (Oracle 중 현재 레이블과 일치하는 문서 탐색)
            matching_oracles = [doc for doc in oracle_docs if doc.get('label') == q_label]
            
            # 매칭되는 문서가 없으면 전체 Oracle 중 유사도 기준 탐색
            if not matching_oracles:
                matching_oracles = oracle_docs
                
            if not matching_oracles:
                print(f"⚠️ 경고: 질문 {q_idx}에 매칭할 Oracle 문서가 존재하지 않습니다. 건너뜁니다.")
                continue
                
            # 1. Golden context 찾기 (유사도가 가장 높은 문서 하나 선택)
            cand_texts = [d.get('text', '') for d in matching_oracles]
            sims = self.calculate_similarity(query, cand_texts)
            best_idx = np.argmax(sims) if len(sims) > 0 else 0
            golden_doc = matching_oracles[best_idx]
            
            # 2. Hard Negatives 구성 (유사 도메인 또는 동일 레이블에서 Golden을 제외한 다른 문서들)
            # Golden 본인은 제외
            other_oracles = [d for d in matching_oracles if d.get('text') != golden_doc.get('text')]
            if not other_oracles:
                # 핑퐁 도메인 매핑으로 확장
                mapped_label = self.hard_negative_mapping.get(q_label)
                other_oracles = [d for d in oracle_docs if d.get('label') == mapped_label]
            if not other_oracles:
                other_oracles = [d for d in oracle_docs if d.get('text') != golden_doc.get('text')]
                
            hard_negatives = self.get_hard_negatives(query, other_oracles, count=hard_neg_count)
            
            # 3. Easy Negative 구성
            easy_negative = self.get_easy_negative(distractor_docs)
            
            # RAFT 포맷 조립
            raft_sample = {
                "query": query,
                "golden": {
                    "label": golden_doc.get("label", ""),
                    "text": golden_doc.get("text", ""),
                    "source": golden_doc.get("source", ""),
                    "title": golden_doc.get("title", "")
                },
                "hard_negatives": hard_negatives,
                "easy_negative": easy_negative if easy_negative else None,
                "answer": answer
            }
            
            raft_dataset.append(raft_sample)
            success_count += 1
            
        # JSONL 저장
        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in raft_dataset:
                json.dump(sample, f, ensure_ascii=False)
                f.write('\n')
                
        print(f"RAFT 데이터 빌드 완료: {success_count}개 샘플을 '{output_file}'에 생성했습니다.")
        return raft_dataset
