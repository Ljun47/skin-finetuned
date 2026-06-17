import os
import gc
from dotenv import load_dotenv

def init_environment():
    """
    W&B 로깅을 완전히 비활성화하고, 프로젝트 루트의 .env를 찾아 로드하며,
    CUDA 캐시를 정리하여 초기 학습 환경을 조성합니다.
    """
    # W&B 비활성화 설정
    os.environ['WANDB_MODE'] = 'disabled'
    
    # 💡 환경 변수(.env) 자동 스캔 및 로드
    # 현재 파일(training/utils/env_helper.py) 기준으로 상위 폴더들을 탐색하며 .env 로드
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_env = os.path.join(current_dir, "../../.env")
    root_env_alt = os.path.join(current_dir, "../../../.env")
    backend_env = os.path.join(current_dir, "../../service/backend/.env")
    local_env = os.path.join(current_dir, ".env")
    
    loaded = False
    for env_path in [local_env, backend_env, root_env, root_env_alt]:
        if os.path.exists(env_path):
            load_dotenv(env_path)
            print(f"✅ 환경 변수 파일 로드 성공: {env_path}")
            loaded = True
            break
            
    if not loaded:
        # 시스템 기본 환경변수 백업 로딩
        load_dotenv()
        print("⚠️  특정 경로의 .env를 찾지 못하여 기본 load_dotenv()를 실행했습니다.")

def clear_cuda_cache():
    """
    PyTorch CUDA 메모리 캐시를 해제하고 가비지 컬렉션을 수집합니다.
    """
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        print(f"🧹 GPU 메모리 캐시 정리 완료: 할당됨={allocated:.2f} GB, 예약됨={reserved:.2f} GB")
    else:
        print("ℹ️ CUDA를 사용할 수 없어 GPU 캐시 정리를 건너뜁니다.")
