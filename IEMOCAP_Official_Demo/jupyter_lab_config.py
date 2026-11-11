"""Windows 현장 실행에 사용할 프로젝트 전용 Jupyter 설정임."""
from pathlib import Path
from urllib.parse import quote

# 실행 위치와 관계없이 이 설정 파일이 있는 패키지를 기준으로 노트북을 엶.
package_root = Path(__file__).resolve().parent
c = get_config()
c.ServerApp.root_dir = str(package_root)
c.ServerApp.default_url = "/lab/tree/" + quote("현장시연.ipynb")
c.ServerApp.ip = "127.0.0.1"
c.ServerApp.open_browser = True
# 한글 사용자 폴더의 임시 리다이렉트 파일 대신 로컬 HTTP 주소로 브라우저를 엶.
# Jupyter의 기본 토큰 인증은 그대로 사용함.
c.ServerApp.use_redirect_file = False

# 설치 파일이 이 프로젝트의 .venv 안에 등록한 커널만 선택 목록에 표시함.
c.KernelSpecManager.allowed_kernelspecs = {"iemocap-demo-windows"}
c.KernelSpecManager.ensure_native_kernel = False
c.MultiKernelManager.default_kernel_name = "iemocap-demo-windows"
