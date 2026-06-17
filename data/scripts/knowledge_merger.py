import json
import os
from collections import Counter

class KnowledgeMerger:
    def __init__(self, oracle_labels=None, label_mapping=None, source_mapping=None):
        # Oracle로 분류될 label 목록
        self.oracle_labels = oracle_labels or {
            "Psoriasis", "Seborrheic Dermatitis", "Seborrheic",
            "Rosacea", "Acne", "Atopic Dermatitis", "Atopic"
        }
        
        # 변경할 label 매핑
        self.label_mapping = label_mapping or {
            "Seborrheic": "Seborrheic Dermatitis",
            "Seborrheic dermatitis": "Seborrheic Dermatitis",
            "Seborrheic Dermatitis": "Seborrheic Dermatitis",
            "Atopic": "Atopic Dermatitis",
            "Atopic dermatitis": "Atopic Dermatitis",
            "Atopic Dermatitis": "Atopic Dermatitis"
        }
        
        # 변경할 source 매핑
        self.source_mapping = source_mapping or {
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

    def process_and_merge_jsonl_files(self, file1, file2, file3, output_file):
        """
        3개의 JSONL 파일을 읽어서 공통 구조로 변환 후 하나로 합치는 함수
        """
        all_documents = []

        # 각 파일 처리
        for file_path in [file1, file2, file3]:
            if not os.path.exists(file_path):
                print(f"⚠️ 경고: {file_path} 파일이 존재하지 않아 합치기에서 제외합니다.")
                continue

            print(f"처리 중: {file_path}")
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

                        # 6. document_type 처리 (임시 저장)
                        processed_doc['_original_doc_type'] = doc.get('document_type', None)

                        all_documents.append(processed_doc)
                        doc_count += 1

                    except json.JSONDecodeError as e:
                        print(f"  줄 {line_num + 1} 파싱 에러: {e}")

                print(f"  {doc_count}개 문서 처리 완료")

        # 7. 모든 문서에 대해 document_type 최종 처리
        print("document_type 최종 처리 중...")
        for doc in all_documents:
            if doc['_original_doc_type'] is not None:
                doc['document_type'] = doc['_original_doc_type']
            else:
                if doc['label'] in self.oracle_labels:
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

        print(f"총 {len(all_documents)}개 문서를 '{output_file}'에 저장했습니다.")
        self.print_statistics(all_documents)
        return all_documents

    def update_labels_in_jsonl(self, input_file, output_file):
        """
        JSONL 파일의 label 값을 정규화 매핑하여 저장하는 함수
        """
        updated_count = 0
        total_count = 0
        label_changes = {}

        if not os.path.exists(input_file):
            raise FileNotFoundError(f"입력 파일 {input_file}이 존재하지 않습니다.")

        print(f"Label 정규화 처리 중: {input_file}")

        with open(input_file, 'r', encoding='utf-8') as f_in, \
             open(output_file, 'w', encoding='utf-8') as f_out:

            for line_num, line in enumerate(f_in):
                if not line.strip():
                    continue

                try:
                    doc = json.loads(line)
                    total_count += 1

                    if 'label' in doc:
                        old_label = doc['label']

                        if old_label in self.label_mapping:
                            new_label = self.label_mapping[old_label]

                            if old_label != new_label:
                                doc['label'] = new_label
                                updated_count += 1

                                if old_label not in label_changes:
                                    label_changes[old_label] = 0
                                label_changes[old_label] += 1

                    json.dump(doc, f_out, ensure_ascii=False)
                    f_out.write('\n')

                except json.JSONDecodeError as e:
                    print(f"줄 {line_num + 1} 파싱 에러: {e}")
                    f_out.write(line)

        print(f"총 처리된 문서 수: {total_count}, 업데이트된 문서 수: {updated_count}")
        if label_changes:
            print("변경 내역:")
            for old_label, count in sorted(label_changes.items()):
                new_label = self.label_mapping[old_label]
                print(f"  '{old_label}' → '{new_label}': {count}개")
        return updated_count

    def update_sources_in_jsonl(self, input_file, output_file):
        """
        JSONL 파일의 source 값을 정규화 매핑하여 저장하는 함수
        """
        updated_count = 0
        total_count = 0
        source_changes = {}

        if not os.path.exists(input_file):
            raise FileNotFoundError(f"입력 파일 {input_file}이 존재하지 않습니다.")

        print(f"Source 정규화 처리 중: {input_file}")

        with open(input_file, 'r', encoding='utf-8') as f_in, \
             open(output_file, 'w', encoding='utf-8') as f_out:

            for line_num, line in enumerate(f_in):
                if not line.strip():
                    continue

                try:
                    doc = json.loads(line)
                    total_count += 1

                    if 'source' in doc:
                        old_source = doc['source']

                        if old_source in self.source_mapping:
                            new_source = self.source_mapping[old_source]
                            
                            if old_source != new_source:
                                doc['source'] = new_source
                                updated_count += 1

                                if old_source not in source_changes:
                                    source_changes[old_source] = 0
                                source_changes[old_source] += 1

                    json.dump(doc, f_out, ensure_ascii=False)
                    f_out.write('\n')

                except json.JSONDecodeError as e:
                    print(f"줄 {line_num + 1} 파싱 에러: {e}")
                    f_out.write(line)

        print(f"총 처리된 문서 수: {total_count}, 업데이트된 문서 수: {updated_count}")
        if source_changes:
            print("변경 내역:")
            for old_source, count in sorted(source_changes.items()):
                new_source = self.source_mapping[old_source]
                print(f"  '{old_source}' → '{new_source}': {count}개")
        return updated_count

    def update_normal_document_type(self, input_file, output_file):
        """
        label이 'Normal'인 문서의 document_type을 'oracle'로 변경하는 함수
        """
        updated_count = 0
        total_count = 0

        if not os.path.exists(input_file):
            raise FileNotFoundError(f"입력 파일 {input_file}이 존재하지 않습니다.")

        print(f"Normal 문서 타입 처리 중: {input_file}")

        with open(input_file, 'r', encoding='utf-8') as f_in, \
             open(output_file, 'w', encoding='utf-8') as f_out:

            for line_num, line in enumerate(f_in):
                if not line.strip():
                    continue

                try:
                    doc = json.loads(line)
                    total_count += 1

                    if doc.get('label') == 'Normal':
                        if doc.get('document_type') != 'oracle':
                            doc['document_type'] = 'oracle'
                            updated_count += 1

                    json.dump(doc, f_out, ensure_ascii=False)
                    f_out.write('\n')

                except json.JSONDecodeError as e:
                    print(f"줄 {line_num + 1} 파싱 에러: {e}")
                    f_out.write(line)

        print(f"총 처리된 문서 수: {total_count}, 'Normal'➔'oracle' 변경 문서 수: {updated_count}")
        return updated_count

    def remove_short_texts(self, input_file, output_file, max_length=50):
        """
        너무 짧은 텍스트를 제거하여 정제하는 함수
        """
        total_count = 0
        removed_count = 0

        if not os.path.exists(input_file):
            raise FileNotFoundError(f"입력 파일 {input_file}이 존재하지 않습니다.")

        print(f"짧은 텍스트 제거 처리 중 ({max_length}글자 이하): {input_file}")

        with open(input_file, 'r', encoding='utf-8') as f_in, \
             open(output_file, 'w', encoding='utf-8') as f_out:

            for line_num, line in enumerate(f_in):
                if not line.strip():
                    continue

                try:
                    doc = json.loads(line)
                    total_count += 1
                    
                    text = doc.get('text', '')
                    if len(text) <= max_length:
                        removed_count += 1
                        continue

                    json.dump(doc, f_out, ensure_ascii=False)
                    f_out.write('\n')

                except json.JSONDecodeError as e:
                    print(f"줄 {line_num + 1} 파싱 에러: {e}")
                    f_out.write(line)

        print(f"총 문서 수: {total_count}, 제거된 짧은 문서 수: {removed_count}, 최종 문서 수: {total_count - removed_count}")
        return removed_count

    def count_labels(self, jsonl_file):
        """
        JSONL 파일 내의 label 종류와 문서 개수를 카운팅하고 출력하는 유틸리티
        """
        labels = []
        if not os.path.exists(jsonl_file):
            print(f"⚠️ 파일 없음: {jsonl_file}")
            return {}

        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        doc = json.loads(line)
                        if 'label' in doc:
                            labels.append(doc['label'])
                    except:
                        pass

        label_counts = Counter(labels)
        print(f"\n=== Label 분포 분석 ({os.path.basename(jsonl_file)}) ===")
        for label, count in label_counts.most_common():
            percentage = (count / len(labels) * 100) if len(labels) > 0 else 0
            print(f"  {label}: {count}개 ({percentage:.1f}%)")
        return label_counts

    def print_statistics(self, documents):
        label_counts = {}
        doc_type_counts = {'oracle': 0, 'distractor': 0, 'none': 0}

        for doc in documents:
            label = doc.get('label', 'Unknown')
            label_counts[label] = label_counts.get(label, 0) + 1

            doc_type = doc.get('document_type', 'none')
            if doc_type in doc_type_counts:
                doc_type_counts[doc_type] += 1
            else:
                doc_type_counts[doc_type] = doc_type_counts.get(doc_type, 0) + 1

        print("\n=== 통계 ===")
        print("Label별 문서 수:")
        for label, count in sorted(label_counts.items()):
            print(f"  {label}: {count}개")
        print("Document type별 문서 수:")
        for doc_type, count in doc_type_counts.items():
            print(f"  {doc_type}: {count}개")
