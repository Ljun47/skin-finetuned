import argparse
import os
import sys

# 모듈 경로를 검색 경로에 추가하여 동일 폴더 내 모듈 임포트 가능하도록 설정
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from knowledge_merger import KnowledgeMerger
from raft_dataset_builder import RaftDatasetBuilder

def run_pipeline(args):
    merger = KnowledgeMerger()
    builder = RaftDatasetBuilder()

    # 임시 및 최종 파일 경로 설정
    raw_merged = "merged_output_raw.jsonl"
    label_fixed = "merged_output_label_fixed.jsonl"
    source_fixed = "merged_output_source_fixed.jsonl"
    normal_fixed = "merged_output_normal_fixed.jsonl"
    
    print("🚀 [1단계] raw 지식 정보 JSONL 파일 3개 병합 시작...")
    merger.process_and_merge_jsonl_files(args.file1, args.file2, args.file3, raw_merged)
    
    print("\n🚀 [2단계] 질환 Label 명칭 정규화 처리 시작...")
    merger.update_labels_in_jsonl(raw_merged, label_fixed)
    merger.count_labels(label_fixed)

    print("\n🚀 [3단계] 출처(Source) 표기 명칭 정규화 처리 시작...")
    merger.update_sources_in_jsonl(label_fixed, source_fixed)

    print("\n🚀 [4단계] Normal(정상 피부) 도메인 문서 타입 설정 시작...")
    merger.update_normal_document_type(source_fixed, normal_fixed)

    print("\n🚀 [5단계] 불필요하게 짧은 단답형 텍스트 지문 필터링 정제 시작...")
    merger.remove_short_texts(normal_fixed, args.kb_output, max_length=args.min_len)
    
    # 정제된 최종 지식베이스 통계 출력
    merger.count_labels(args.kb_output)

    # RAFT 데이터셋 생성 파트
    if args.questions and os.path.exists(args.questions):
        print(f"\n🚀 [6단계] 질문 템플릿과 정제된 지식 베이스를 융합하여 RAFT 데이터셋 빌드 시작...")
        builder.generate_raft_samples(
            questions_file=args.questions,
            candidate_file=args.kb_output,
            output_file=args.raft_output,
            hard_neg_count=args.hard_neg_count
        )
    else:
        print("\n⚠️ [알림] 질문 데이터 경로(--questions)가 입력되지 않았거나 존재하지 않아 RAFT 데이터셋 생성은 건너뜁니다.")

    # 임시 중간 정제 파일들 삭제 (옵션)
    if not args.keep_temp:
        print("\n🧹 임시 중간 생성 파일 제거 중...")
        for temp_file in [raw_merged, label_fixed, source_fixed, normal_fixed]:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        print("✅ 임시 파일 제거 완료!")

    print("\n🎉 모든 데이터 정제 및 병합 전처리 파이프라인 완수!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="피부 질환 도메인 지식 병합 및 RAFT 전처리 파이프라인 CLI")
    
    # 원천 데이터 경로 인자
    parser.add_argument("--file1", type=str, default="data/sample_data/final_knowledge_chunks_for_embedding.jsonl", help="원천 지식 chunks 파일 1")
    parser.add_argument("--file2", type=str, default="data/sample_data/kb_docs_chunked.jsonl", help="원천 지식 chunks 파일 2")
    parser.add_argument("--file3", type=str, default="data/sample_data/raft_dataset.jsonl", help="원천 지식 chunks 파일 3")
    
    # 전처리 및 결과 아웃풋 인자
    parser.add_argument("--kb_output", type=str, default="data/sample_data/cleaned_knowledge_base.jsonl", help="정제 완료된 최종 지식베이스 경로")
    parser.add_argument("--min_len", type=int, default=50, help="지문 정제 기준 최소 텍스트 글자 수")
    
    # RAFT 생성 관련 인자
    parser.add_argument("--questions", type=str, default="data/sample_data/raft_golden_samples.txt", help="골든 샘플 질문 텍스트 파일 경로")
    parser.add_argument("--raft_output", type=str, default="data/sample_data/raft_train_dataset_final.jsonl", help="최종 출력될 RAFT 학습 JSONL 경로")
    parser.add_argument("--hard_neg_count", type=int, default=4, help="샘플당 할당할 Hard Negative 문맥 개수")
    
    # 임시 파일 보관 여부
    parser.add_argument("--keep_temp", action="store_true", help="중간 파이프라인 임시 생성 파일(Label/Source 정규화 중간 결과) 삭제 방지")

    args = parser.parse_args()
    
    # 상대경로 실행에 맞게 파일 존재 여부 확인 후 입력 인자 조정
    # 기본 경로 설정값 확인 및 자동 조정
    if not os.path.exists(args.file1):
        # /content/ 등 또는 로컬 루트 실행 대비 탐색
        for prefix in ["", "../../", "./"]:
            candidate = os.path.join(prefix, args.file1)
            if os.path.exists(candidate):
                args.file1 = candidate
                break
                
    # 나머지 기본 매개변수 경로들도 보정
    for attr in ["file2", "file3", "questions"]:
        val = getattr(args, attr)
        if not os.path.exists(val):
            for prefix in ["", "../../", "./"]:
                candidate = os.path.join(prefix, val)
                if os.path.exists(candidate):
                    setattr(args, attr, candidate)
                    break

    run_pipeline(args)
