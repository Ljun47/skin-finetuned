from huggingface_hub import HfApi

api = HfApi()

# 1. 업로드할 로컬의 1차 SFT 최종 어댑터 경로
LOCAL_FOLDER = "/Users/jun/Downloads/취업/기본 프로젝트/1차 파인튜닝/1차_파인튜닝_어댑터(최종모델)/checkpoint-300"

# 2. 허깅페이스에 생성할 모델 레포지토리 이름 (계정명/원하는이름)
REPO_ID = "jun47/skin-llava-sft-lora"

print("1. 허깅페이스에 1차 SFT 어댑터 전용 레포지토리 생성 중...")
api.create_repo(
    repo_id=REPO_ID,
    repo_type="model",
    private=False  # 외부 시연 및 포트폴리오용이므로 Public 설정
)

print(f"2. {LOCAL_FOLDER}의 가중치를 {REPO_ID}로 업로드 시작...")
api.upload_folder(
    folder_path=LOCAL_FOLDER,
    repo_id=REPO_ID,
    repo_type="model"
)
print(f"🎉 기본 프로젝트 1차 SFT 어댑터 업로드 완료!")
print(f"👉 확인 주소: https://huggingface.co/{REPO_ID}")
