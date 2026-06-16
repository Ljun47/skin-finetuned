# ==================================================
# [Code Cell]
# ==================================================
import json

def process_and_merge_jsonl_files(file1, file2, file3, output_file):
    """
    3개의 JSONL 파일을 읽어서 공통 구조로 변환 후 하나로 합치는 함수
    """
    all_documents = []

    # Oracle로 분류될 label 목록
    oracle_labels = {
        "Psoriasis", "Seborrheic Dermatitis", "Seborrheic",
        "Rosacea", "Acne", "Atopic Dermatitis", "Atopic"
    }

    # 각 파일 처리
    for file_path in [file1, file2, file3]:
        print(f"\n처리 중: {file_path}")

        with open(file_path, 'r', encoding='utf-8') as f:
            doc_count = 0
            for line_num, line in enumerate(f):
                if not line.strip():
                    continue

                try:
                    doc = json.loads(line)
                    processed_doc = {}

                    # 1. source 유지
                    processed_doc['source'] = doc.get('source', '')

                    # 2. label 첫 글자 대문자로 변경
                    label = doc.get('label', '')
                    processed_doc['label'] = label.capitalize()

                    # 3. title 처리
                    if 'section' in doc:  # 1,2번 파일
                        processed_doc['title'] = doc['section']
                    elif 'title' in doc:  # 3번 파일
                        processed_doc['title'] = doc['title']
                    else:
                        processed_doc['title'] = ''

                    # 4. text 처리
                    if 'text' in doc:  # 1,2번 파일
                        processed_doc['text'] = doc['text']
                    elif 'content' in doc:  # 3번 파일
                        processed_doc['text'] = doc['content']
                    else:
                        processed_doc['text'] = ''

                    # 5. text_length 처리
                    if 'text_length' in doc:
                        processed_doc['text_length'] = doc['text_length']
                    else:
                        processed_doc['text_length'] = len(processed_doc['text'])

                    # 6. document_type 처리 (임시로 저장, 나중에 처리)
                    processed_doc['_original_doc_type'] = doc.get('document_type', None)

                    all_documents.append(processed_doc)
                    doc_count += 1

                except json.JSONDecodeError as e:
                    print(f"  줄 {line_num + 1} 파싱 에러: {e}")

            print(f"  {doc_count}개 문서 처리 완료")

    # 7. 모든 문서에 대해 document_type 최종 처리
    print("\ndocument_type 최종 처리 중...")
    for doc in all_documents:
        if doc['_original_doc_type'] is not None:
            # 기존 document_type이 있으면 유지
            doc['document_type'] = doc['_original_doc_type']
        else:
            # 없으면 label에 따라 설정
            if doc['label'] in oracle_labels:
                doc['document_type'] = 'oracle'
            else:
                doc['document_type'] = 'distractor'

        # 임시 필드 제거
        del doc['_original_doc_type']

    # 8. 결과를 하나의 JSONL 파일로 저장
    with open(output_file, 'w', encoding='utf-8') as f:
        for doc in all_documents:
            json.dump(doc, f, ensure_ascii=False)
            f.write('\n')

    print(f"\n총 {len(all_documents)}개 문서를 '{output_file}'에 저장했습니다.")

    # 통계 출력
    print("\n=== 통계 ===")
    label_counts = {}
    doc_type_counts = {'oracle': 0, 'distractor': 0, 'none': 0}

    for doc in all_documents:
        # Label별 카운트
        label = doc['label']
        label_counts[label] = label_counts.get(label, 0) + 1

        # Document type별 카운트
        doc_type = doc.get('document_type', 'none')
        if doc_type in doc_type_counts:
            doc_type_counts[doc_type] += 1
        else:
            doc_type_counts[doc_type] = doc_type_counts.get(doc_type, 0) + 1

    print("\nLabel별 문서 수:")
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count}개")

    print("\nDocument type별 문서 수:")
    for doc_type, count in doc_type_counts.items():
        print(f"  {doc_type}: {count}개")

    return all_documents

# 사용 예시
if __name__ == "__main__":
    # 3개의 JSONL 파일 경로 지정
    file1 = "/content/final_knowledge_chunks_for_embedding.jsonl"  # 첫 번째 JSONL 파일
    file2 = "/content/kb_docs_chunked.jsonl"  # 두 번째 JSONL 파일
    file3 = "/content/raft_dataset.jsonl"  # 세 번째 JSONL 파일 (labeled_output.jsonl)

    # 통합된 출력 파일
    output = "merged_output.jsonl"

    # 처리 실행
    merged_data = process_and_merge_jsonl_files(file1, file2, file3, output)

    # 결과 샘플 확인
    print("\n=== 결과 샘플 (처음 3개) ===")
    for i, doc in enumerate(merged_data[:3]):
        print(f"\n문서 {i+1}:")
        for key, value in doc.items():
            if key == 'text':
                print(f"  {key}: {value[:50]}...")
            else:
                print(f"  {key}: {value}")



# ==================================================
# [Code Cell]
# ==================================================
import json

def update_labels_in_jsonl(input_file, output_file):
    """
    JSONL 파일의 label 값을 업데이트하는 함수
    - Seborrheic → Seborrheic Dermatitis
    - Seborrheic dermatitis → Seborrheic Dermatitis
    - Atopic → Atopic Dermatitis
    - Atopic dermatitis → Atopic Dermatitis
    """
    # 변경할 label 매핑 (대소문자 구분 없이 처리하기 위해 lower case도 포함)
    label_mapping = {
        "Seborrheic": "Seborrheic Dermatitis",
        "Seborrheic dermatitis": "Seborrheic Dermatitis",
        "Seborrheic Dermatitis": "Seborrheic Dermatitis",  # 이미 올바른 형식이어도 통일성을 위해
        "Atopic": "Atopic Dermatitis",
        "Atopic dermatitis": "Atopic Dermatitis",
        "Atopic Dermatitis": "Atopic Dermatitis"  # 이미 올바른 형식이어도 통일성을 위해
    }

    updated_count = 0
    total_count = 0
    label_changes = {}

    print(f"파일 처리 중: {input_file}")

    # 읽고 쓰기
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:

        for line_num, line in enumerate(f_in):
            if not line.strip():
                continue

            try:
                doc = json.loads(line)
                total_count += 1

                # label이 있는 경우
                if 'label' in doc:
                    old_label = doc['label']

                    # 변경이 필요한 경우
                    if old_label in label_mapping:
                        new_label = label_mapping[old_label]

                        # 실제로 변경이 일어나는 경우만 카운트
                        if old_label != new_label:
                            doc['label'] = new_label
                            updated_count += 1

                            # 변경 내역 기록
                            if old_label not in label_changes:
                                label_changes[old_label] = 0
                            label_changes[old_label] += 1

                # 수정된(또는 원본) 문서를 출력 파일에 쓰기
                json.dump(doc, f_out, ensure_ascii=False)
                f_out.write('\n')

            except json.JSONDecodeError as e:
                print(f"줄 {line_num + 1} 파싱 에러: {e}")
                # 파싱 에러가 난 줄은 원본 그대로 복사
                f_out.write(line)

    # 결과 출력
    print(f"\n=== Label 업데이트 결과 ===")
    print(f"총 처리된 문서 수: {total_count}")
    print(f"업데이트된 문서 수: {updated_count}")

    if label_changes:
        print(f"\n변경 내역:")
        for old_label, count in sorted(label_changes.items()):
            new_label = label_mapping[old_label]
            print(f"  '{old_label}' → '{new_label}': {count}개")
    else:
        print("\n변경된 label이 없습니다.")

    print(f"\n결과가 '{output_file}'에 저장되었습니다.")

    return updated_count

def check_labels_before_after(input_file, output_file):
    """
    변경 전후의 label을 비교하는 함수
    """
    def get_label_counts(file_path):
        label_counts = {}
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        doc = json.loads(line)
                        if 'label' in doc:
                            label = doc['label']
                            label_counts[label] = label_counts.get(label, 0) + 1
                    except:
                        pass
        return label_counts

    print("\n=== Label 변경 전후 비교 ===")

    # 변경 전
    before_counts = get_label_counts(input_file)
    print("\n변경 전 label 목록:")
    for label in sorted(before_counts.keys()):
        print(f"  {label}: {before_counts[label]}개")

    # 변경 후
    after_counts = get_label_counts(output_file)
    print("\n변경 후 label 목록:")
    for label in sorted(after_counts.keys()):
        print(f"  {label}: {after_counts[label]}개")

    # 변경 확인
    target_labels = ["Seborrheic", "Seborrheic dermatitis", "Atopic", "Atopic dermatitis"]
    remaining = [label for label in target_labels if label in after_counts]

    if remaining:
        print(f"\n⚠️ 아직 변경되지 않은 label: {remaining}")
    else:
        print("\n✅ 모든 대상 label이 성공적으로 통합되었습니다.")

# 사용 예시
if __name__ == "__main__":
    # 입력 파일과 출력 파일 지정
    input_jsonl = "merged_output.jsonl"  # 원본 파일
    output_jsonl = "updated_labels_final.jsonl"  # 업데이트된 파일

    # Label 업데이트 실행
    update_labels_in_jsonl(input_jsonl, output_jsonl)

    # 변경 전후 비교
    check_labels_before_after(input_jsonl, output_jsonl)



# ==================================================
# [Code Cell]
# ==================================================
import json
from collections import Counter

def analyze_labels_in_jsonl(file_path):
    """
    JSONL 파일을 읽어서 label의 종류와 각 label별 개수를 분석하는 함수
    """
    labels = []
    total_docs = 0
    error_count = 0
    no_label_count = 0

    print(f"분석 중: {file_path}\n")

    # JSONL 파일 읽기
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            if not line.strip():
                continue

            try:
                doc = json.loads(line)
                total_docs += 1

                # label 키가 있는지 확인
                if 'label' in doc:
                    labels.append(doc['label'])
                else:
                    no_label_count += 1

            except json.JSONDecodeError as e:
                error_count += 1
                print(f"줄 {line_num + 1} 파싱 에러: {e}")

    # label별 개수 카운트
    label_counts = Counter(labels)

    # 결과 출력
    print("=== Label 분석 결과 ===")
    print(f"총 문서 수: {total_docs}")
    print(f"label이 있는 문서 수: {len(labels)}")
    print(f"label이 없는 문서 수: {no_label_count}")
    print(f"파싱 에러 수: {error_count}")

    print(f"\n총 {len(label_counts)}개의 고유한 label이 발견되었습니다.")

    # label별 개수 출력 (많은 순서대로)
    print("\n=== Label별 문서 개수 (많은 순) ===")
    for label, count in label_counts.most_common():
        percentage = (count / len(labels) * 100) if len(labels) > 0 else 0
        print(f"{label}: {count}개 ({percentage:.1f}%)")

    # 알파벳 순서로도 출력
    print("\n=== Label별 문서 개수 (알파벳 순) ===")
    for label in sorted(label_counts.keys()):
        count = label_counts[label]
        percentage = (count / len(labels) * 100) if len(labels) > 0 else 0
        print(f"{label}: {count}개 ({percentage:.1f}%)")

    return label_counts

def save_label_analysis(file_path, output_path):
    """
    Label 분석 결과를 텍스트 파일로 저장하는 함수
    """
    label_counts = analyze_labels_in_jsonl(file_path)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"Label 분석 결과 - {file_path}\n")
        f.write("=" * 50 + "\n\n")

        f.write(f"총 고유 label 수: {len(label_counts)}\n")
        f.write(f"총 문서 수: {sum(label_counts.values())}\n\n")

        f.write("Label별 개수 (많은 순):\n")
        f.write("-" * 30 + "\n")
        for label, count in label_counts.most_common():
            percentage = (count / sum(label_counts.values()) * 100)
            f.write(f"{label}: {count}개 ({percentage:.1f}%)\n")

        f.write("\n\nLabel별 개수 (알파벳 순):\n")
        f.write("-" * 30 + "\n")
        for label in sorted(label_counts.keys()):
            count = label_counts[label]
            percentage = (count / sum(label_counts.values()) * 100)
            f.write(f"{label}: {count}개 ({percentage:.1f}%)\n")

    print(f"\n분석 결과가 '{output_path}'에 저장되었습니다.")

# 사용 예시
if __name__ == "__main__":
    # 분석할 JSONL 파일 경로
    jsonl_file = "merged_output.jsonl"  # 실제 파일명으로 변경

    # 기본 분석
    label_stats = analyze_labels_in_jsonl(jsonl_file)

    # 분석 결과를 파일로 저장하고 싶다면
    # save_label_analysis(jsonl_file, "label_analysis.txt")



# ==================================================
# [Code Cell]
# ==================================================
import json
from collections import Counter

def analyze_sources_in_jsonl(file_path):
    """
    JSONL 파일을 읽어서 source의 종류와 각 source별 개수를 분석하는 함수
    """
    sources = []
    total_docs = 0
    error_count = 0
    no_source_count = 0

    print(f"분석 중: {file_path}\n")

    # JSONL 파일 읽기
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            if not line.strip():
                continue

            try:
                doc = json.loads(line)
                total_docs += 1

                # source 키가 있는지 확인
                if 'source' in doc:
                    sources.append(doc['source'])
                else:
                    no_source_count += 1

            except json.JSONDecodeError as e:
                error_count += 1
                print(f"줄 {line_num + 1} 파싱 에러: {e}")

    # source별 개수 카운트
    source_counts = Counter(sources)

    # 결과 출력
    print("=== Source 분석 결과 ===")
    print(f"총 문서 수: {total_docs}")
    print(f"source가 있는 문서 수: {len(sources)}")
    print(f"source가 없는 문서 수: {no_source_count}")
    print(f"파싱 에러 수: {error_count}")

    print(f"\n총 {len(source_counts)}개의 고유한 source가 발견되었습니다.")

    # source별 개수 출력 (많은 순서대로)
    print("\n=== Source별 문서 개수 (많은 순) ===")
    for source, count in source_counts.most_common():
        percentage = (count / len(sources) * 100) if len(sources) > 0 else 0
        print(f"{source}: {count}개 ({percentage:.1f}%)")

    # 알파벳 순서로도 출력
    print("\n=== Source별 문서 개수 (알파벳 순) ===")
    for source in sorted(source_counts.keys()):
        count = source_counts[source]
        percentage = (count / len(sources) * 100) if len(sources) > 0 else 0
        print(f"{source}: {count}개 ({percentage:.1f}%)")

    return source_counts

def analyze_source_by_label(file_path):
    """
    Label별로 source 분포를 분석하는 함수
    """
    label_source_data = {}

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    doc = json.loads(line)
                    label = doc.get('label', 'NO_LABEL')
                    source = doc.get('source', 'NO_SOURCE')

                    if label not in label_source_data:
                        label_source_data[label] = []
                    label_source_data[label].append(source)
                except:
                    pass

    print("\n=== Label별 Source 분포 ===")
    for label in sorted(label_source_data.keys()):
        sources = label_source_data[label]
        source_counts = Counter(sources)
        print(f"\n{label} (총 {len(sources)}개 문서):")
        for source, count in source_counts.most_common():
            percentage = (count / len(sources) * 100)
            print(f"  {source}: {count}개 ({percentage:.1f}%)")

def save_source_analysis(file_path, output_path):
    """
    Source 분석 결과를 텍스트 파일로 저장하는 함수
    """
    source_counts = analyze_sources_in_jsonl(file_path)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"Source 분석 결과 - {file_path}\n")
        f.write("=" * 50 + "\n\n")

        f.write(f"총 고유 source 수: {len(source_counts)}\n")
        f.write(f"총 문서 수: {sum(source_counts.values())}\n\n")

        f.write("Source별 개수 (많은 순):\n")
        f.write("-" * 30 + "\n")
        for source, count in source_counts.most_common():
            percentage = (count / sum(source_counts.values()) * 100)
            f.write(f"{source}: {count}개 ({percentage:.1f}%)\n")

        f.write("\n\nSource별 개수 (알파벳 순):\n")
        f.write("-" * 30 + "\n")
        for source in sorted(source_counts.keys()):
            count = source_counts[source]
            percentage = (count / sum(source_counts.values()) * 100)
            f.write(f"{source}: {count}개 ({percentage:.1f}%)\n")

    print(f"\n분석 결과가 '{output_path}'에 저장되었습니다.")

# 사용 예시
if __name__ == "__main__":
    # 분석할 JSONL 파일 경로
    jsonl_file = "/content/updated_sources_final.jsonl"  # 실제 파일명으로 변경

    # 기본 source 분석
    source_stats = analyze_sources_in_jsonl(jsonl_file)

    # Label별 source 분포 분석
    analyze_source_by_label(jsonl_file)

    # 분석 결과를 파일로 저장하고 싶다면
    # save_source_analysis(jsonl_file, "source_analysis.txt")



# ==================================================
# [Code Cell]
# ==================================================
import json

def update_sources_in_jsonl(input_file, output_file):
    """
    JSONL 파일의 source 값을 업데이트하는 함수
    """
    # 변경할 source 매핑
    source_mapping = {
        "서울아산병원": "아산병원",
        "Asan": "아산병원",
        "대한피부과학회 아토피피부염 가이드라인": "대한피부과학회",
        "대한피부과학회 여드름 가이드라인": "대한피부과학회",
        "분당서울대학교병원": "서울대학교병원",
        "분당서울대병원": "서울대학교병원",
        "서울대학교 어린이병원": "서울대학교병원",
        "서울대학교병원 의학정보": "서울대학교병원",
        "연세대학교 세브란스병원": "연세대 세브란스병원"
    }

    updated_count = 0
    total_count = 0
    source_changes = {}

    print(f"파일 처리 중: {input_file}")

    # 읽고 쓰기
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:

        for line_num, line in enumerate(f_in):
            if not line.strip():
                continue

            try:
                doc = json.loads(line)
                total_count += 1

                # source가 있는 경우
                if 'source' in doc:
                    old_source = doc['source']

                    # 변경이 필요한 경우
                    if old_source in source_mapping:
                        new_source = source_mapping[old_source]
                        doc['source'] = new_source
                        updated_count += 1

                        # 변경 내역 기록
                        if old_source not in source_changes:
                            source_changes[old_source] = 0
                        source_changes[old_source] += 1

                # 수정된(또는 원본) 문서를 출력 파일에 쓰기
                json.dump(doc, f_out, ensure_ascii=False)
                f_out.write('\n')

            except json.JSONDecodeError as e:
                print(f"줄 {line_num + 1} 파싱 에러: {e}")
                f_out.write(line)

    # 결과 출력
    print(f"\n=== Source 업데이트 결과 ===")
    print(f"총 처리된 문서 수: {total_count}")
    print(f"업데이트된 문서 수: {updated_count}")

    if source_changes:
        print(f"\n변경 내역:")
        for old_source, count in sorted(source_changes.items()):
            new_source = source_mapping[old_source]
            print(f"  '{old_source}' → '{new_source}': {count}개")
    else:
        print("\n변경된 source가 없습니다.")

    print(f"\n결과가 '{output_file}'에 저장되었습니다.")

    return updated_count



# ==================================================
# [Code Cell]
# ==================================================
import json

def view_documents_by_source(file_path, target_source, max_display=None):
    """
    특정 source에 해당하는 모든 문서를 보는 함수

    Args:
        file_path: JSONL 파일 경로
        target_source: 찾고자 하는 source 값
        max_display: 출력할 최대 문서 수 (None이면 모두 출력)
    """
    matching_docs = []

    print(f"\n=== Source '{target_source}'에 해당하는 문서 검색 ===\n")

    # JSONL 파일 읽기
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            if not line.strip():
                continue

            try:
                doc = json.loads(line)
                if doc.get('source') == target_source:
                    matching_docs.append({
                        'line_num': line_num + 1,
                        'doc': doc
                    })
            except json.JSONDecodeError:
                pass

    # 결과 출력
    print(f"총 {len(matching_docs)}개의 문서를 찾았습니다.")

    if matching_docs:
        # 출력할 문서 수 결정
        display_count = len(matching_docs) if max_display is None else min(max_display, len(matching_docs))

        print(f"\n{'모든' if max_display is None else f'처음 {display_count}개'} 문서 출력:\n")
        print("="*80)

        for i, item in enumerate(matching_docs[:display_count]):
            doc = item['doc']
            print(f"\n[문서 {i+1}] (줄 번호: {item['line_num']})")
            print(f"Source: {doc.get('source', 'N/A')}")
            print(f"Label: {doc.get('label', 'N/A')}")
            print(f"Title: {doc.get('title', doc.get('section', 'N/A'))}")
            print(f"Document Type: {doc.get('document_type', 'N/A')}")
            print(f"Text Length: {doc.get('text_length', 'N/A')}")
            print(f"Text: {doc.get('text', 'N/A')[:200]}{'...' if len(doc.get('text', '')) > 200 else ''}")
            print("-"*80)

        if max_display and len(matching_docs) > max_display:
            print(f"\n... 그리고 {len(matching_docs) - max_display}개 더 있습니다.")

        # 통계 정보
        print(f"\n=== 통계 정보 ===")

        # Label별 분포
        labels = [item['doc'].get('label', 'Unknown') for item in matching_docs]
        label_counts = {}
        for label in labels:
            label_counts[label] = label_counts.get(label, 0) + 1

        print(f"\nLabel별 분포:")
        for label, count in sorted(label_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {label}: {count}개 ({count/len(matching_docs)*100:.1f}%)")

        # Document Type별 분포
        doc_types = [item['doc'].get('document_type', 'Unknown') for item in matching_docs]
        doc_type_counts = {}
        for doc_type in doc_types:
            doc_type_counts[doc_type] = doc_type_counts.get(doc_type, 0) + 1

        print(f"\nDocument Type별 분포:")
        for doc_type, count in sorted(doc_type_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {doc_type}: {count}개 ({count/len(matching_docs)*100:.1f}%)")

        # 평균 텍스트 길이
        text_lengths = [item['doc'].get('text_length', 0) for item in matching_docs]
        if text_lengths:
            avg_length = sum(text_lengths) / len(text_lengths)
            print(f"\n평균 텍스트 길이: {avg_length:.1f} 글자")

    else:
        print(f"\nSource '{target_source}'에 해당하는 문서를 찾을 수 없습니다.")

    return matching_docs

def save_source_documents_to_file(file_path, target_source, output_file):
    """
    특정 source의 문서들을 별도 파일로 저장하는 함수
    """
    print(f"\nSource '{target_source}'의 문서를 '{output_file}'에 저장 중...")

    matching_docs = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    doc = json.loads(line)
                    if doc.get('source') == target_source:
                        matching_docs.append(doc)
                except:
                    pass

    if matching_docs:
        with open(output_file, 'w', encoding='utf-8') as f:
            for doc in matching_docs:
                json.dump(doc, f, ensure_ascii=False)
                f.write('\n')

        print(f"{len(matching_docs)}개의 문서를 저장했습니다.")
    else:
        print("해당 source의 문서를 찾을 수 없습니다.")

# 사용 예시
if __name__ == "__main__":
    # 1. Source 업데이트 사용 예시
    input_file = "updated_labels_final.jsonl"
    output_file = "updated_sources_final.jsonl"

    update_sources_in_jsonl(input_file, output_file)

    # 모든 문서를 보려면 max_display=None
    # view_documents_by_source(output_file, "서울대학교병원", max_display=None)

    # 특정 source의 문서를 별도 파일로 저장
    # save_source_documents_to_file(output_file, "대한피부과학회", "dermatology_society_docs.jsonl")



# ==================================================
# [Code Cell]
# ==================================================
import json

def update_normal_document_type(input_file, output_file):
    """
    label이 'Normal'인 문서의 document_type을 'oracle'로 변경하는 함수
    """
    updated_count = 0
    total_count = 0
    normal_count = 0
    already_oracle = 0

    print(f"파일 처리 중: {input_file}")

    # 읽고 쓰기
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:

        for line_num, line in enumerate(f_in):
            if not line.strip():
                continue

            try:
                doc = json.loads(line)
                total_count += 1

                # label이 'Normal'인 경우
                if doc.get('label') == 'Normal':
                    normal_count += 1

                    # document_type이 이미 oracle인지 확인
                    if doc.get('document_type') == 'oracle':
                        already_oracle += 1
                    else:
                        # oracle로 변경
                        old_type = doc.get('document_type', 'None')
                        doc['document_type'] = 'oracle'
                        updated_count += 1

                        if line_num < 3:  # 처음 몇 개 샘플 출력
                            print(f"\n변경 예시 {line_num + 1}:")
                            print(f"  Label: {doc.get('label')}")
                            print(f"  Document type: '{old_type}' → 'oracle'")

                # 수정된(또는 원본) 문서를 출력 파일에 쓰기
                json.dump(doc, f_out, ensure_ascii=False)
                f_out.write('\n')

            except json.JSONDecodeError as e:
                print(f"줄 {line_num + 1} 파싱 에러: {e}")
                f_out.write(line)

    # 결과 출력
    print(f"\n=== Document Type 업데이트 결과 ===")
    print(f"총 처리된 문서 수: {total_count}")
    print(f"Label이 'Normal'인 문서 수: {normal_count}")
    print(f"  - 이미 oracle인 문서: {already_oracle}")
    print(f"  - oracle로 변경된 문서: {updated_count}")

    print(f"\n결과가 '{output_file}'에 저장되었습니다.")

    return updated_count

def verify_normal_documents(file_path):
    """
    업데이트 후 Normal label 문서들의 document_type 확인
    """
    normal_docs = []
    doc_type_counts = {}

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    doc = json.loads(line)
                    if doc.get('label') == 'Normal':
                        normal_docs.append(doc)
                        doc_type = doc.get('document_type', 'None')
                        doc_type_counts[doc_type] = doc_type_counts.get(doc_type, 0) + 1
                except:
                    pass

    print(f"\n=== Normal Label 문서 검증 ===")
    print(f"Total Normal 문서: {len(normal_docs)}개")
    print("\nDocument Type 분포:")
    for doc_type, count in sorted(doc_type_counts.items()):
        print(f"  {doc_type}: {count}개 ({count/len(normal_docs)*100:.1f}%)")

    # oracle이 아닌 문서가 있는지 확인
    non_oracle = sum(count for doc_type, count in doc_type_counts.items() if doc_type != 'oracle')
    if non_oracle > 0:
        print(f"\n⚠️ 경고: {non_oracle}개의 Normal 문서가 아직 oracle이 아닙니다!")
    else:
        print("\n✅ 모든 Normal 문서가 oracle로 설정되었습니다.")

    # 샘플 출력
    print("\n처음 3개 Normal 문서 샘플:")
    for i, doc in enumerate(normal_docs[:3]):
        print(f"\n문서 {i+1}:")
        print(f"  Label: {doc.get('label')}")
        print(f"  Document Type: {doc.get('document_type')}")
        print(f"  Source: {doc.get('source')}")
        print(f"  Text: {doc.get('text', '')[:100]}...")

# 사용 예시
if __name__ == "__main__":
    # 입력/출력 파일 지정
    input_file = "updated_sources_final.jsonl"  # 실제 파일명으로 변경
    output_file = "updated_normal_oracle.jsonl"

    # Normal label의 document_type을 oracle로 변경
    update_normal_document_type(input_file, output_file)

    # 변경 결과 확인
    print("\n" + "="*50)
    verify_normal_documents(output_file)



# ==================================================
# [Code Cell]
# ==================================================
# 예시: '아산병원' source의 문서 보기 (처음 5개만)
view_documents_by_source(output_file, "청소년 건강행태온라인조사", max_display=5)



# ==================================================
# [Code Cell]
# ==================================================
import json

def extract_specific_labels(input_file, output_file, target_labels):
    """
    JSONL 파일에서 특정 label의 문서만 추출하여 텍스트 파일로 저장

    Args:
        input_file: 입력 JSONL 파일 경로
        output_file: 출력 텍스트 파일 경로
        target_labels: 추출할 label 리스트
    """
    extracted_docs = []

    # JSONL 파일 읽기
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            if not line.strip():
                continue

            try:
                doc = json.loads(line)
                # 특정 label인 경우만 추출
                if doc.get('label') in target_labels:
                    extracted_docs.append(doc)
            except json.JSONDecodeError as e:
                print(f"줄 {line_num + 1} 파싱 에러: {e}")

    # 텍스트 파일로 저장
    with open(output_file, 'w', encoding='utf-8') as f:
        # Label별로 구분하여 저장
        for label in target_labels:
            label_docs = [doc for doc in extracted_docs if doc.get('label') == label]

            if label_docs:
                f.write(f"{'='*80}\n")
                f.write(f"LABEL: {label} (총 {len(label_docs)}개 문서)\n")
                f.write(f"{'='*80}\n\n")

                for i, doc in enumerate(label_docs, 1):
                    f.write(f"[문서 {i}]\n")
                    f.write(f"Label: {doc.get('label', 'N/A')}\n")
                    f.write(f"Source: {doc.get('source', 'N/A')}\n")
                    f.write(f"Title: {doc.get('title', 'N/A')}\n")
                    f.write(f"Document Type: {doc.get('document_type', 'N/A')}\n")
                    f.write(f"Text Length: {doc.get('text_length', 'N/A')}\n")
                    f.write(f"Text: {doc.get('text', 'N/A')}\n")
                    f.write(f"{'-'*80}\n\n")

    print(f"추출 완료:")
    for label in target_labels:
        count = len([doc for doc in extracted_docs if doc.get('label') == label])
        print(f"  {label}: {count}개")
    print(f"총 {len(extracted_docs)}개 문서를 '{output_file}'에 저장했습니다.")

    return extracted_docs

def extract_text_only(input_file, output_file, target_labels):
    """
    특정 label의 text 내용만 추출하여 저장 (쿼리 생성용)
    """
    texts_by_label = {label: [] for label in target_labels}

    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    doc = json.loads(line)
                    if doc.get('label') in target_labels:
                        texts_by_label[doc['label']].append({
                            'title': doc.get('title', 'No title'),
                            'text': doc.get('text', ''),
                            'source': doc.get('source', 'Unknown')
                        })
                except:
                    pass

    # 텍스트만 저장
    with open(output_file, 'w', encoding='utf-8') as f:
        for label in target_labels:
            texts = texts_by_label[label]
            if texts:
                f.write(f"\n{'='*80}\n")
                f.write(f"LABEL: {label} (총 {len(texts)}개)\n")
                f.write(f"{'='*80}\n\n")

                for i, item in enumerate(texts, 1):
                    f.write(f"[{label} - 문서 {i}] {item['title']} (출처: {item['source']})\n")
                    f.write(f"{item['text']}\n")
                    f.write(f"\n{'-'*40}\n\n")

    print(f"\nText만 추출 완료:")
    for label, texts in texts_by_label.items():
        print(f"  {label}: {len(texts)}개")

# 사용 예시
if __name__ == "__main__":
    # 입력 파일과 추출할 label 설정
    input_jsonl = "raft_dataset_final.jsonl"  # 실제 파일명으로 변경
    output_txt = "atopic_seborrheic_docs.txt"
    output_text_only = "atopic_seborrheic_texts.txt"

    target_labels = ["Atopic Dermatitis", "Seborrheic Dermatitis"]

    # 전체 정보 추출
    extract_specific_labels(input_jsonl, output_txt, target_labels)

    # 텍스트만 추출 (쿼리 생성용)
    extract_text_only(input_jsonl, output_text_only, target_labels)



"""
##**학습데이터 제작**
"""


# ==================================================
# [Code Cell]
# ==================================================
import json
import re

def parse_text_file_debug(file_path):
    """
    텍스트 파일을 파싱하여 문서와 질문들을 추출하는 함수 (디버깅 버전)
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    print(f"파일 크기: {len(content)} 문자")
    print(f"첫 500자: {content[:500]}")
    print("-" * 50)

    documents = []

    # 더 간단한 방법: 청크별로 분리
    # JSON과 질문이 연속으로 나타나는 패턴을 찾기

    # 방법 1: JSON 문자열 패턴으로 찾기
    json_pattern = r'\{[^}]+?"document_type":\s*"oracle"\s*\}'
    json_matches = re.finditer(json_pattern, content, re.DOTALL)

    json_positions = []
    for match in json_matches:
        json_str = match.group()
        try:
            doc_data = json.loads(json_str)
            json_positions.append((match.start(), match.end(), doc_data))
            print(f"JSON 찾음: {doc_data['label']} - {doc_data['title']}")
        except json.JSONDecodeError as e:
            print(f"JSON 파싱 실패: {e}")

    print(f"\n찾은 JSON 문서 수: {len(json_positions)}")

    # 각 JSON 문서 다음에 나오는 질문들 찾기
    for i, (start, end, doc_data) in enumerate(json_positions):
        # 현재 JSON 이후부터 다음 JSON까지의 텍스트
        if i < len(json_positions) - 1:
            next_start = json_positions[i + 1][0]
            section = content[end:next_start]
        else:
            section = content[end:]

        # 질문 찾기 (숫자. 로 시작하는 라인)
        questions = []
        lines = section.split('\n')

        for line in lines:
            line = line.strip()
            match = re.match(r'^(\d+)\.\s+(.+)$', line)
            if match:
                questions.append(match.group(2))

        if questions:
            print(f"{doc_data['label']}에 대한 질문 {len(questions)}개 찾음")
            documents.append({
                'doc_data': doc_data,
                'questions': questions
            })

    return documents

def parse_text_file_alternative(file_path):
    """
    대체 파싱 방법: 청크 단위로 처리
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    documents = []

    # **청크 1**, **청크 2** 등으로 구분된 경우
    chunks = re.split(r'\*\*청크\s+\d+\*\*', content)

    for chunk in chunks[1:]:  # 첫 번째는 빈 문자열이므로 제외
        if not chunk.strip():
            continue

        # JSON 찾기
        json_match = re.search(r'\{[^}]+\}', chunk, re.DOTALL)
        if not json_match:
            continue

        try:
            doc_data = json.loads(json_match.group())
        except json.JSONDecodeError:
            continue

        # 질문 찾기
        questions = []
        lines = chunk.split('\n')

        for line in lines:
            line = line.strip()
            match = re.match(r'^(\d+)\.\s+(.+)$', line)
            if match:
                questions.append(match.group(2))

        if questions:
            documents.append({
                'doc_data': doc_data,
                'questions': questions
            })

    return documents

def create_raft_data_from_documents(documents):
    """
    파싱된 문서들로부터 RAFT 데이터 생성
    """
    all_raft_data = []
    doc_counter = {}

    for doc_item in documents:
        doc_data = doc_item['doc_data']
        questions = doc_item['questions']

        label = doc_data['label']
        label_short = label.lower().replace(' ', '_')

        if label not in doc_counter:
            doc_counter[label] = 1
        else:
            doc_counter[label] += 1

        doc_id = f"{label_short}_doc_{doc_counter[label]:03d}"

        for idx, question in enumerate(questions, 1):
            question_id = f"{label_short}_doc{doc_counter[label]:03d}_q{idx:03d}"

            raft_item = {
                "question_id": question_id,
                "question": question,
                "golden_doc": {
                    "doc_id": doc_id,
                    "text": doc_data['text'],
                    "label": doc_data['label'],
                    "source": doc_data['source'],
                    "document_type": doc_data['document_type'],
                    "title": doc_data.get('title', ''),
                    "text_length": doc_data.get('text_length', len(doc_data['text']))
                }
            }

            all_raft_data.append(raft_item)

    return all_raft_data

def save_as_jsonl(raft_data, output_file):
    """
    RAFT 데이터를 JSONL 형식으로 저장
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in raft_data:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')

def process_text_file(input_file, output_file="raft_golden_samples.jsonl"):
    """
    텍스트 파일을 처리하여 RAFT 데이터 생성
    """
    print(f"텍스트 파일 처리 중: {input_file}")

    # 먼저 디버깅 버전으로 시도
    documents = parse_text_file_debug(input_file)

    # 문서가 없으면 대체 방법 시도
    if not documents:
        print("\n대체 파싱 방법 시도 중...")
        documents = parse_text_file_alternative(input_file)

    print(f"\n최종 파싱된 문서 수: {len(documents)}")

    if not documents:
        print("문서를 찾을 수 없습니다. 파일 형식을 확인해주세요.")
        return []

    # RAFT 데이터 생성
    raft_data = create_raft_data_from_documents(documents)

    # JSONL로 저장
    save_as_jsonl(raft_data, output_file)

    print(f"\n총 {len(raft_data)}개의 RAFT 데이터 생성 완료")
    print(f"출력 파일: {output_file}")

    return raft_data

# 간단한 수동 파싱 함수 (파일 형식이 특수한 경우)
def manual_parse_example():
    """
    수동으로 데이터를 입력하는 예시
    """
    doc_json = {"source": "MSD Manual", "label": "Acne", "title": "여드름의 원인",
                "text": "여드름은 모낭(모발이 자라는 피부의 구멍)의 염증을...",
                "text_length": 670, "document_type": "oracle"}

    questions = [
        "여드름이 생기는 가장 기본적인 이유가 궁금해요. 호르몬, 피지, 세균이 서로 어떻게 작용해서 여드름이 되는 건가요?",
        "블랙헤드랑 패립종(화이트헤드)은 뭐가 다른 거예요?",
        # ... 나머지 질문들
    ]

    documents = [{
        'doc_data': doc_json,
        'questions': questions
    }]

    return create_raft_data_from_documents(documents)

if __name__ == "__main__":
    input_file = "Q01.txt"
    output_file = "raft_golden_samples.jsonl"

    # 파일 처리 시도
    raft_data = process_text_file(input_file, output_file)

    # 결과가 없으면 파일 내용 일부를 직접 확인
    if not raft_data:
        print("\n파일 내용을 직접 확인해주세요.")
        print("예상 형식:")
        print("1. JSON 객체가 한 줄로 되어 있어야 함")
        print("2. 질문은 '숫자. 질문내용' 형식이어야 함")
        print("3. JSON과 질문 사이에 적절한 구분이 있어야 함")



# ==================================================
# [Code Cell]
# ==================================================
import json
import re

def parse_alternating_format(file_path):
    """
    JSON과 질문 리스트가 번갈아 나타나는 형식을 파싱
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    documents = []

    # JSON 객체를 찾는 정규표현식
    json_pattern = r'\{[^}]+?"document_type":\s*"[^"]+"\s*\}'

    # 모든 JSON 객체의 위치 찾기
    json_matches = list(re.finditer(json_pattern, content, re.DOTALL))

    print(f"찾은 JSON 객체 수: {len(json_matches)}")

    for i, match in enumerate(json_matches):
        json_str = match.group()

        # JSON 파싱
        try:
            doc_data = json.loads(json_str)
            print(f"\nJSON {i+1}: {doc_data['label']} - {doc_data['title']}")
        except json.JSONDecodeError as e:
            print(f"JSON 파싱 에러: {e}")
            continue

        # 현재 JSON 다음부터 다음 JSON까지의 텍스트에서 질문 찾기
        start_pos = match.end()
        end_pos = json_matches[i+1].start() if i+1 < len(json_matches) else len(content)

        question_section = content[start_pos:end_pos]

        # 질문 찾기 (숫자. 로 시작하는 줄)
        questions = []
        lines = question_section.split('\n')

        for line in lines:
            line = line.strip()
            # 1. 또는 1) 형식의 질문 찾기
            match = re.match(r'^(\d+)[.)\s]+(.+)$', line)
            if match:
                question_text = match.group(2).strip()
                questions.append(question_text)

        if questions:
            print(f"  -> {len(questions)}개 질문 찾음")
            documents.append({
                'doc_data': doc_data,
                'questions': questions
            })

    return documents

def create_raft_data_from_documents(documents):
    """
    파싱된 문서들로부터 RAFT 데이터 생성
    """
    all_raft_data = []
    doc_counter = {}

    for doc_idx, doc_item in enumerate(documents):
        doc_data = doc_item['doc_data']
        questions = doc_item['questions']

        label = doc_data['label']
        label_short = label.lower().replace(' ', '_')

        # 라벨별 문서 카운터
        if label not in doc_counter:
            doc_counter[label] = 1
        else:
            doc_counter[label] += 1

        # 문서 ID 생성
        doc_id = f"{label_short}_doc_{doc_counter[label]:03d}"

        # 각 질문에 대해 RAFT 아이템 생성
        for q_idx, question in enumerate(questions, 1):
            question_id = f"{label_short}_doc{doc_counter[label]:03d}_q{q_idx:03d}"

            raft_item = {
                "question_id": question_id,
                "question": question,
                "golden_doc": {
                    "doc_id": doc_id,
                    "text": doc_data['text'],
                    "label": doc_data['label'],
                    "source": doc_data['source'],
                    "document_type": doc_data['document_type'],
                    "title": doc_data.get('title', ''),
                    "text_length": doc_data.get('text_length', len(doc_data['text']))
                }
            }

            all_raft_data.append(raft_item)

    return all_raft_data

def save_as_jsonl(raft_data, output_file):
    """
    RAFT 데이터를 JSONL 형식으로 저장
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in raft_data:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')

def process_alternating_format_file(input_file, output_file="raft_golden_samples.jsonl"):
    """
    번갈아 나타나는 형식의 텍스트 파일 처리
    """
    print(f"텍스트 파일 처리 중: {input_file}")
    print("형식: JSON 객체와 질문 리스트가 번갈아 나타남")
    print("-" * 50)

    # 문서 파싱
    documents = parse_alternating_format(input_file)

    print(f"\n총 파싱된 문서 수: {len(documents)}")

    if not documents:
        print("문서를 찾을 수 없습니다.")
        return []

    # RAFT 데이터 생성
    raft_data = create_raft_data_from_documents(documents)

    # JSONL로 저장
    save_as_jsonl(raft_data, output_file)

    # 통계 출력
    print(f"\n=== 처리 완료 ===")
    print(f"총 {len(raft_data)}개의 RAFT 데이터 생성")
    print(f"출력 파일: {output_file}")

    # 라벨별 통계
    label_stats = {}
    doc_stats = {}

    for item in raft_data:
        label = item['golden_doc']['label']
        doc_id = item['golden_doc']['doc_id']

        if label not in label_stats:
            label_stats[label] = 0
            doc_stats[label] = set()

        label_stats[label] += 1
        doc_stats[label].add(doc_id)

    print("\n라벨별 통계:")
    for label in sorted(label_stats.keys()):
        print(f"  {label}:")
        print(f"    - 문서 수: {len(doc_stats[label])}")
        print(f"    - 질문 수: {label_stats[label]}")
        print(f"    - 문서당 평균 질문 수: {label_stats[label] / len(doc_stats[label]):.1f}")

    # 샘플 출력
    if raft_data:
        print("\n=== 생성된 데이터 샘플 ===")
        sample = raft_data[0]
        print(json.dumps(sample, ensure_ascii=False, indent=2))

    return raft_data

# 디버깅을 위한 함수
def check_file_format(file_path, lines_to_check=50):
    """
    파일 형식 확인용 함수
    """
    print(f"파일 형식 확인: {file_path}")
    print("-" * 50)

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    print(f"총 줄 수: {len(lines)}")
    print(f"처음 {lines_to_check}줄:")

    for i, line in enumerate(lines[:lines_to_check]):
        if line.strip():
            print(f"{i+1}: {line[:100].strip()}{'...' if len(line) > 100 else ''}")

if __name__ == "__main__":
    input_file = "Q02.txt"  # 실제 파일명으로 변경
    output_file = "raft_golden_samples02.jsonl"

    # 파일 형식 확인 (디버깅용)
    # check_file_format(input_file)

    # 파일 처리
    raft_data = process_alternating_format_file(input_file, output_file)



# ==================================================
# [Code Cell]
# ==================================================
import json
import re

def load_questions_data(questions_file_path):
    """질문 데이터 텍스트 파일을 로드하고 파싱합니다."""
    with open(questions_file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    questions_data = {}

    # 전체 내용에서 "document_N" 패턴을 모두 찾기
    # 더 유연한 패턴 사용
    doc_pattern = r'"document_(\d+)":\s*\{((?:[^{}]|{[^{}]*})*)\}'
    matches = re.findall(doc_pattern, content, re.DOTALL)

    print(f"정규표현식으로 찾은 문서 수: {len(matches)}")

    # 매치되지 않은 경우를 위한 대안 방법
    if len(matches) < 87:  # 예상보다 적으면
        print("대안 파싱 방법 사용...")

        # 수동으로 "document_" 위치 찾기
        doc_positions = []
        for match in re.finditer(r'"document_(\d+)":', content):
            doc_num = int(match.group(1))
            start_pos = match.start()
            doc_positions.append((doc_num, start_pos))

        # 위치 정렬
        doc_positions.sort(key=lambda x: x[1])

        questions_data = {}
        doc_counter = 1
        current_category = 'atopic'

        for i, (local_doc_num, start_pos) in enumerate(doc_positions):
            # 다음 문서의 시작 위치 찾기
            if i + 1 < len(doc_positions):
                end_pos = doc_positions[i + 1][1]
            else:
                end_pos = len(content)

            # 해당 문서의 전체 내용 추출
            doc_content = content[start_pos:end_pos]

            # 카테고리 판단 (아토피는 1-51, 지루성은 1-36이지만 두 번째 그룹)
            # 새로운 JSON 객체가 시작되는지 확인
            if '},\n{' in content[max(0, start_pos-100):start_pos]:
                current_category = 'seborrheic'

            # text 추출
            text_match = re.search(r'"text":\s*"(.*?)",\s*"queries"', doc_content, re.DOTALL)
            if text_match:
                text = text_match.group(1)
            else:
                continue

            # queries 추출
            queries_match = re.search(r'"queries":\s*\[(.*?)\]\s*\}', doc_content, re.DOTALL)
            if queries_match:
                queries_content = queries_match.group(1)
                # 개별 query 추출 (더 정확한 패턴)
                queries = []
                for query_match in re.finditer(r'"([^"]+)"', queries_content):
                    queries.append(query_match.group(1))
            else:
                queries = []

            questions_data[f"document_{doc_counter}"] = {
                'text': text,
                'queries': queries,
                'category': current_category,
                'local_doc_num': local_doc_num
            }
            doc_counter += 1

            # 아토피에서 지루성으로 넘어가는 지점 체크 (더 정확한 방법)
            if current_category == 'atopic' and local_doc_num == 51:
                current_category = 'seborrheic'

        return questions_data

    # 원래 방법이 성공한 경우
    doc_counter = 1
    current_category = 'atopic'

    for local_doc_num_str, doc_content in matches:
        local_doc_num = int(local_doc_num_str)

        # 카테고리 전환 점검
        if current_category == 'atopic' and local_doc_num == 1 and doc_counter > 51:
            current_category = 'seborrheic'

        # text 추출
        text_match = re.search(r'"text":\s*"(.*?)",', doc_content, re.DOTALL)
        if text_match:
            text = text_match.group(1)
        else:
            continue

        # queries 추출
        queries_match = re.search(r'"queries":\s*\[(.*?)\]', doc_content, re.DOTALL)
        if queries_match:
            queries_content = queries_match.group(1)
            queries = re.findall(r'"([^"]*)"', queries_content)
        else:
            queries = []

        questions_data[f"document_{doc_counter}"] = {
            'text': text,
            'queries': queries,
            'category': current_category,
            'local_doc_num': local_doc_num
        }
        doc_counter += 1

    return questions_data

def load_metadata(metadata_file_path):
    """메타데이터 파일을 파싱합니다."""
    with open(metadata_file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    metadata_docs = []

    # LABEL로 섹션 분리
    sections = re.split(r'={80,}\nLABEL: ([^(]+)', content)[1:]

    for i in range(0, len(sections), 2):
        if i + 1 < len(sections):
            label = sections[i].strip()
            section_content = sections[i + 1]

            # 카테고리 판단
            category = 'atopic' if 'Atopic' in label else 'seborrheic'

            # 각 문서별로 분리
            docs = re.split(r'\[문서 (\d+)\]', section_content)[1:]

            for j in range(0, len(docs), 2):
                if j + 1 < len(docs):
                    doc_num = docs[j].strip()
                    doc_content = docs[j + 1]

                    doc_info = {
                        'label': label,
                        'doc_number': int(doc_num),
                        'category': category
                    }

                    lines = doc_content.strip().split('\n')
                    text_lines = []
                    collecting_text = False

                    for line in lines:
                        if line.startswith('Label:'):
                            doc_info['label'] = line.split(':', 1)[1].strip()
                        elif line.startswith('Source:'):
                            doc_info['source'] = line.split(':', 1)[1].strip()
                        elif line.startswith('Title:'):
                            doc_info['title'] = line.split(':', 1)[1].strip()
                        elif line.startswith('Document Type:'):
                            doc_info['document_type'] = line.split(':', 1)[1].strip()
                        elif line.startswith('Text Length:'):
                            doc_info['text_length'] = int(line.split(':', 1)[1].strip())
                        elif line.startswith('Text:'):
                            text_lines.append(line.split(':', 1)[1].strip())
                            collecting_text = True
                        elif collecting_text and line.startswith('---'):
                            collecting_text = False
                        elif collecting_text:
                            text_lines.append(line.strip())

                    if text_lines:
                        doc_info['text'] = ' '.join(text_lines).strip()

                    metadata_docs.append(doc_info)

    return metadata_docs

def match_text_with_metadata(questions_data, metadata_docs):
    """질문 데이터의 텍스트와 메타데이터를 매칭합니다."""
    matched_data = []

    for doc_key, doc_data in questions_data.items():
        question_text = doc_data['text']
        queries = doc_data['queries']
        category = doc_data['category']
        local_doc_num = doc_data['local_doc_num']

        # 같은 카테고리의 메타데이터에서 매칭
        matched_metadata = None
        for meta_doc in metadata_docs:
            if (meta_doc['category'] == category and
                meta_doc['doc_number'] == local_doc_num and
                'text' in meta_doc):
                matched_metadata = meta_doc
                break

        # 텍스트 기반 매칭 (번호 매칭이 실패한 경우)
        if not matched_metadata:
            for meta_doc in metadata_docs:
                if (meta_doc['category'] == category and 'text' in meta_doc):
                    if (question_text in meta_doc['text'] or
                        meta_doc['text'] in question_text or
                        re.sub(r'\s+', '', question_text) in re.sub(r'\s+', '', meta_doc['text'])):
                        matched_metadata = meta_doc
                        break

        if matched_metadata:
            doc_number = int(doc_key.split('_')[1])

            matched_data.append({
                'doc_number': doc_number,
                'local_doc_num': local_doc_num,
                'category': category,
                'original_text': question_text,
                'queries': queries,
                'metadata': matched_metadata
            })
        else:
            print(f"Warning: No metadata found for {doc_key} (category: {category}, local_num: {local_doc_num})")

    return matched_data

def generate_golden_samples(matched_data, output_file_path):
    """골든 샘플을 생성하고 JSONL 파일로 저장합니다."""

    with open(output_file_path, 'w', encoding='utf-8') as f:
        for doc_data in matched_data:
            doc_number = doc_data['doc_number']
            local_doc_num = doc_data['local_doc_num']
            category = doc_data['category']
            queries = doc_data['queries']
            metadata = doc_data['metadata']

            for i, question in enumerate(queries, 1):
                question_id = f"{category}_doc{doc_number:03d}_q{i:03d}"
                doc_id = f"{category}_doc_{doc_number:03d}"

                golden_sample = {
                    "question_id": question_id,
                    "question": question,
                    "golden_doc": {
                        "doc_id": doc_id,
                        "text": metadata['text'],
                        "label": metadata['label'],
                        "source": metadata.get('source', ''),
                        "document_type": metadata.get('document_type', ''),
                        "title": metadata.get('title', ''),
                        "text_length": metadata.get('text_length', len(metadata['text']))
                    }
                }

                f.write(json.dumps(golden_sample, ensure_ascii=False) + '\n')

def main():
    """메인 실행 함수"""
    questions_file_path = "Q03.txt"
    metadata_file_path = "atopic_seborrheic_docs.txt"
    output_file_path = "raft_golden_samples03.jsonl"

    print("질문 데이터를 로딩 중...")
    questions_data = load_questions_data(questions_file_path)
    print(f"로드된 질문 문서 수: {len(questions_data)}")

    atopic_count = sum(1 for doc in questions_data.values() if doc['category'] == 'atopic')
    seborrheic_count = sum(1 for doc in questions_data.values() if doc['category'] == 'seborrheic')
    print(f"아토피 문서: {atopic_count}개, 지루성 문서: {seborrheic_count}개")

    print("메타데이터를 파싱 중...")
    metadata_docs = load_metadata(metadata_file_path)
    print(f"로드된 메타데이터 문서 수: {len(metadata_docs)}")

    print("텍스트와 메타데이터를 매칭 중...")
    matched_data = match_text_with_metadata(questions_data, metadata_docs)
    print(f"매칭된 문서 수: {len(matched_data)}")

    print("골든 샘플을 생성 중...")
    generate_golden_samples(matched_data, output_file_path)

    total_samples = sum(len(doc['queries']) for doc in matched_data)
    print(f"작업 완료! {len(matched_data)} 개 문서에서 총 {total_samples} 개의 골든 샘플이 생성되었습니다.")
    print(f"결과 파일: {output_file_path}")

if __name__ == "__main__":
    main()



# ==================================================
# [Code Cell]
# ==================================================
import json

def fix_seborrheic_doc_numbers(input_file, output_file):
    """seborrheic 문서들의 question_id와 doc_id에서 51을 빼는 함수"""

    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:

        for line in f_in:
            data = json.loads(line.strip())

            # seborrheic 문서만 처리
            if data['question_id'].startswith('seborrheic_'):
                # question_id에서 숫자 추출하고 51 빼기
                question_parts = data['question_id'].split('_')
                doc_part = question_parts[1]  # "doc083" 형태
                doc_num = int(doc_part[3:])   # 083 -> 83
                new_doc_num = doc_num - 51    # 83 - 51 = 32

                # 새로운 question_id 생성
                data['question_id'] = f"seborrheic_doc{new_doc_num:03d}_{question_parts[2]}"

                # doc_id에서도 51 빼기
                doc_id_parts = data['golden_doc']['doc_id'].split('_')
                doc_id_num = int(doc_id_parts[2])  # "seborrheic_doc_083" -> 83
                new_doc_id_num = doc_id_num - 51   # 83 - 51 = 32

                # 새로운 doc_id 생성
                data['golden_doc']['doc_id'] = f"seborrheic_doc_{new_doc_id_num:03d}"

            # 수정된 데이터를 출력 파일에 쓰기
            f_out.write(json.dumps(data, ensure_ascii=False) + '\n')

# 실행
input_file = "raft_golden_samples03.jsonl"      # 원본 파일명
output_file = "raft_golden_samples_fixed03.jsonl"  # 수정된 파일명

fix_seborrheic_doc_numbers(input_file, output_file)
print("seborrheic 문서 번호 수정 완료!")
print(f"수정된 파일: {output_file}")



# ==================================================
# [Code Cell]
# ==================================================
import shutil

def merge_jsonl_files(file1, file2, output_file):
    """두 개의 JSONL 파일을 하나로 합치기"""
    with open(output_file, 'w', encoding='utf-8') as outfile:
        with open(file1, 'r', encoding='utf-8') as infile1:
            shutil.copyfileobj(infile1, outfile)
        with open(file2, 'r', encoding='utf-8') as infile2:
            shutil.copyfileobj(infile2, outfile)

# 실행
merge_jsonl_files("merged.jsonl", "raft_golden_samples_fixed03.jsonl", "raft_golden_samples_final.jsonl")
print("파일 합치기 완료!")



# ==================================================
# [Code Cell]
# ==================================================
import shutil
import json
from collections import Counter

def merge_three_jsonl_files(file1, file2, file3, output_file):
    """세 개의 JSONL 파일을 하나로 합치기"""
    with open(output_file, 'w', encoding='utf-8') as outfile:
        with open(file1, 'r', encoding='utf-8') as infile1:
            shutil.copyfileobj(infile1, outfile)
        with open(file2, 'r', encoding='utf-8') as infile2:
            shutil.copyfileobj(infile2, outfile)
        with open(file3, 'r', encoding='utf-8') as infile3:
            shutil.copyfileobj(infile3, outfile)

def count_labels_in_merged_file(jsonl_file):
    """합쳐진 JSONL 파일에서 golden label들을 카운트하는 함수"""

    label_counts = Counter()

    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            label = data['golden_doc']['label']
            label_counts[label] += 1

    # 결과 출력
    print(f"\n=== 합쳐진 파일 통계 ===")
    print(f"총 샘플 수: {sum(label_counts.values())}")
    print(f"라벨 종류: {len(label_counts)}")
    print("\n라벨별 개수:")
    print("-" * 40)

    for label, count in label_counts.most_common():
        print(f"{label}: {count}개")

    return label_counts

# 실행
merge_three_jsonl_files("raft_golden_samples01.jsonl", "raft_golden_samples02.jsonl", "raft_golden_samples_fixed03.jsonl", "raft_golden_samples_final01.jsonl")
print("파일 합치기 완료!")

# 합쳐진 파일의 라벨 통계 출력
count_labels_in_merged_file("raft_golden_samples_final01.jsonl")



"""
##**네거티브 샘플 제작**
"""


# ==================================================
# [Code Cell]
# ==================================================
import json
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# 도메인 지식 기반 Hard Negative 매핑
HARD_NEGATIVE_MAPPING = {
    "Acne": ["Rosacea", "Seborrheic Dermatitis"],
    "Rosacea": ["Acne", "Seborrheic Dermatitis", "Atopic Dermatitis"],
    "Atopic Dermatitis": ["Seborrheic Dermatitis", "Psoriasis", "Rosacea"],
    "Psoriasis": ["Seborrheic Dermatitis", "Atopic Dermatitis"],
    "Seborrheic Dermatitis": ["Psoriasis", "Atopic Dermatitis", "Acne", "Rosacea"]
}

def load_golden_samples(golden_file):
    """골든 샘플 JSONL 파일 로드"""
    golden_samples = []
    with open(golden_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            golden_samples.append(data)
    return golden_samples

def load_candidate_samples(candidate_file):
    """후보 샘플 JSONL 파일 로드"""
    oracle_samples = []  # Hard negative 용
    distractor_samples = []  # Easy negative 용

    with open(candidate_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            if data['document_type'] == 'oracle':
                oracle_samples.append(data)
            elif data['document_type'] == 'distractor':
                distractor_samples.append(data)

    return oracle_samples, distractor_samples

def calculate_similarity(query_text, candidate_texts):
    """TF-IDF 기반 유사도 계산"""
    texts = [query_text] + candidate_texts
    vectorizer = TfidfVectorizer(stop_words=None, max_features=1000)

    try:
        tfidf_matrix = vectorizer.fit_transform(texts)
        query_vector = tfidf_matrix[0]
        candidate_vectors = tfidf_matrix[1:]

        similarities = cosine_similarity(query_vector, candidate_vectors)[0]
        return similarities
    except:
        # 텍스트가 너무 적거나 문제가 있을 때 랜덤 점수 반환
        return np.random.random(len(candidate_texts))

def get_hard_negatives(golden_sample, oracle_samples):
    """Hard negative 3개 선택"""
    golden_label = golden_sample['golden_doc']['label']
    query_text = golden_sample['question']

    # 도메인 지식으로 후보 라벨 선정
    if golden_label not in HARD_NEGATIVE_MAPPING:
        print(f"Warning: {golden_label} not in mapping, using random selection")
        candidate_labels = list(set([sample['label'] for sample in oracle_samples]))
        candidate_labels = [label for label in candidate_labels if label != golden_label]
        hard_negative_labels = random.sample(candidate_labels, min(3, len(candidate_labels)))
    else:
        hard_negative_labels = HARD_NEGATIVE_MAPPING[golden_label]

    # 해당 라벨의 후보 샘플들 수집
    candidate_samples = []
    for sample in oracle_samples:
        if sample['label'] in hard_negative_labels:
            candidate_samples.append(sample)

    if len(candidate_samples) < 3:
        print(f"Warning: Not enough candidates for {golden_label}, found {len(candidate_samples)}")
        # 부족한 경우 다른 라벨에서도 가져오기
        other_samples = [sample for sample in oracle_samples if sample['label'] != golden_label]
        candidate_samples.extend(other_samples[:3-len(candidate_samples)])

    # 유사도 계산 및 상위 3개 선택
    if len(candidate_samples) > 3:
        candidate_texts = [sample['text'] for sample in candidate_samples]
        similarities = calculate_similarity(query_text, candidate_texts)

        # 유사도 기준으로 정렬하고 상위 3개 선택
        sorted_indices = np.argsort(similarities)[::-1]
        selected_samples = [candidate_samples[i] for i in sorted_indices[:3]]
    else:
        selected_samples = candidate_samples[:3]

    return selected_samples

def get_easy_negative(distractor_samples):
    """Easy negative 1개 랜덤 선택"""
    if distractor_samples:
        return random.choice(distractor_samples)
    else:
        return None

def generate_raft_samples(golden_file, candidate_file, output_file):
    """RAFT 샘플 생성 메인 함수"""

    print("데이터 로딩 중...")
    golden_samples = load_golden_samples(golden_file)
    oracle_samples, distractor_samples = load_candidate_samples(candidate_file)

    print(f"골든 샘플: {len(golden_samples)}개")
    print(f"오라클 샘플: {len(oracle_samples)}개")
    print(f"디스트랙터 샘플: {len(distractor_samples)}개")

    raft_samples = []

    for i, golden_sample in enumerate(golden_samples):
        if i % 100 == 0:
            print(f"처리 중: {i+1}/{len(golden_samples)}")

        # Hard negatives 선택
        hard_negatives = get_hard_negatives(golden_sample, oracle_samples)

        # Easy negative 선택
        easy_negative = get_easy_negative(distractor_samples)

        # RAFT 형태로 구성
        raft_sample = {
            "query": golden_sample['question'],
            "golden": {
                "label": golden_sample['golden_doc']['label'],
                "text": golden_sample['golden_doc']['text'],
                "source": golden_sample['golden_doc']['source'],
                "title": golden_sample['golden_doc']['title'],
                "document_type": golden_sample['golden_doc']['document_type'],
                "text_length": golden_sample['golden_doc']['text_length']
            },
            "hard_negatives": hard_negatives
        }

        # Easy negative가 있으면 추가
        if easy_negative:
            raft_sample["easy_negative"] = easy_negative

        raft_samples.append(raft_sample)

    # 결과 저장
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in raft_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    print(f"완료! {len(raft_samples)}개의 RAFT 샘플이 {output_file}에 저장되었습니다.")

    # 통계 출력
    labels = [sample['golden']['label'] for sample in raft_samples]
    label_counts = {}
    for label in labels:
        label_counts[label] = label_counts.get(label, 0) + 1

    print("\n라벨별 샘플 수:")
    for label, count in label_counts.items():
        print(f"  {label}: {count}개")

def main():
    # 파일 경로 설정
    golden_file = "raft_golden_samples_final01.jsonl"      # 기존 골든 샘플 파일
    candidate_file = "raft_dataset_final.jsonl"  # 모든 후보 샘플 파일 (oracle + distractor)
    output_file = "raft_samples.jsonl"       # 최종 RAFT 샘플 파일

    generate_raft_samples(golden_file, candidate_file, output_file)

if __name__ == "__main__":
    main()



# ==================================================
# [Code Cell]
# ==================================================
import json
from collections import Counter

def count_labels(jsonl_file):
    """JSONL 파일에서 golden label들을 카운트하는 함수"""

    label_counts = Counter()

    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            label = data['golden']['label']
            label_counts[label] += 1

    # 결과 출력
    print(f"총 샘플 수: {sum(label_counts.values())}")
    print(f"라벨 종류: {len(label_counts)}")
    print("\n라벨별 개수:")
    print("-" * 40)

    for label, count in label_counts.most_common():
        print(f"{label}: {count}개")

    return label_counts

# 실행
if __name__ == "__main__":
    jsonl_file = "raft_samples.jsonl"  # 파일명을 실제 파일로 변경
    count_labels(jsonl_file)



"""
##**짧은 텍스트 쳐 내**
"""


# ==================================================
# [Code Cell]
# ==================================================
import json

def show_shortest_texts(jsonl_file, top_n=40):
    """JSONL 파일에서 text_length가 가장 짧은 상위 N개의 text 출력"""

    samples = []

    # 모든 샘플 읽기
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            samples.append({
                'text_length': data['golden']['text_length'],  # golden_doc → golden
                'text': data['golden']['text'],
                'label': data['golden']['label'],
                'query': data['query']  # question_id → query
            })

    # text_length 기준으로 오름차순 정렬
    samples.sort(key=lambda x: x['text_length'])

    # 상위 N개 출력
    print(f"=== text_length가 가장 짧은 상위 {top_n}개 ===\n")

    for i, sample in enumerate(samples[:top_n], 1):
        print(f"{i}. [{sample['label']}] text_length: {sample['text_length']}")
        print(f"   query: {sample['query'][:60]}...")  # 질문 앞부분만 출력
        print(f"   text: {sample['text']}")
        print("-" * 80)

# 실행
if __name__ == "__main__":
    jsonl_file = "raft_samples.jsonl"  # 실제 파일명으로 변경
    show_shortest_texts(jsonl_file, 40)



# ==================================================
# [Code Cell]
# ==================================================
import json

def show_short_texts_unique(jsonl_file, max_length=50):
    """JSONL 파일에서 text_length가 지정값 이하인 텍스트들을 중복 제거하여 출력"""

    seen_texts = set()  # 중복 체크용
    short_samples = []

    # 모든 샘플 읽기
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            text_length = data['golden']['text_length']
            text = data['golden']['text']

            # 길이 조건 확인 및 중복 체크
            if text_length <= max_length and text not in seen_texts:
                seen_texts.add(text)
                short_samples.append({
                    'text_length': text_length,
                    'text': text,
                    'label': data['golden']['label'],
                    'query': data['query']
                })

    # text_length 기준으로 오름차순 정렬
    short_samples.sort(key=lambda x: x['text_length'])

    # 결과 출력
    print(f"=== text_length {max_length} 이하인 고유 텍스트 ({len(short_samples)}개) ===\n")

    for i, sample in enumerate(short_samples, 1):
        print(f"{i}. [{sample['label']}] text_length: {sample['text_length']}")
        print(f"   query: {sample['query'][:60]}...")
        print(f"   text: {sample['text']}")
        print("-" * 80)

# 실행
if __name__ == "__main__":
    jsonl_file = "raft_samples_filtered.jsonl"  # 실제 파일명으로 변경
    show_short_texts_unique(jsonl_file, 60)



# ==================================================
# [Code Cell]
# ==================================================
import json

def remove_short_texts(input_file, output_file, max_length=50):
    """JSONL 파일에서 text_length가 지정값 이하인 샘플들을 제거"""

    removed_count = 0
    kept_count = 0

    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:

        for line in infile:
            data = json.loads(line.strip())
            text_length = data['golden']['text_length']

            # text_length가 50 초과인 것만 유지
            if text_length > max_length:
                outfile.write(json.dumps(data, ensure_ascii=False) + '\n')
                kept_count += 1
            else:
                removed_count += 1

    # 결과 출력
    print(f"=== 필터링 완료 ===")
    print(f"제거된 샘플: {removed_count}개 (text_length <= {max_length})")
    print(f"유지된 샘플: {kept_count}개 (text_length > {max_length})")
    print(f"결과 파일: {output_file}")

# 실행
if __name__ == "__main__":
    input_file = "raft_samples.jsonl"      # 원본 파일
    output_file = "raft_samples_filtered.jsonl"  # 필터링된 파일

    remove_short_texts(input_file, output_file, 50)



# ==================================================
# [Code Cell]
# ==================================================
import json
from collections import Counter

def count_labels(jsonl_file):
    """JSONL 파일에서 golden label들을 카운트하는 함수"""

    label_counts = Counter()

    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            label = data['golden']['label']
            label_counts[label] += 1

    # 결과 출력
    print(f"총 샘플 수: {sum(label_counts.values())}")
    print(f"라벨 종류: {len(label_counts)}")
    print("\n라벨별 개수:")
    print("-" * 40)

    for label, count in label_counts.most_common():
        print(f"{label}: {count}개")

    return label_counts

# 실행
if __name__ == "__main__":
    jsonl_file = "raft_samples_filtered.jsonl"  # 파일명을 실제 파일로 변경
    count_labels(jsonl_file)


