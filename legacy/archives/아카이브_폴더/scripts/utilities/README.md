# 유틸리티 스크립트 (Utility Scripts)

이 디렉토리에는 프로젝트 유지보수 및 분석을 위한 유틸리티 스크립트가 포함되어 있습니다.

## 정리 스크립트

- **cleanup_old_results.py**: 오래된 결과 파일 정리
- **cleanup_unnecessary_files.py**: 불필요한 파일 정리

## 분석 스크립트

- **plot_analytical_comparison.py**: 이론적 비교 그래프 생성
- **plot_psd_verification.py**: PSD 검증 그래프 생성
- **regenerate_plots.py**: 그래프 재생성
- **view_results.py**: 결과 파일 뷰어

## 테스트 및 검증

- **qa_checks.py**: 품질 검사
- **test_parameter_validation.py**: 파라미터 검증 테스트

## 사용법

```bash
# 루트 디렉토리에서 실행
python3 scripts/utilities/cleanup_old_results.py --dry-run
python3 scripts/utilities/plot_analytical_comparison.py
```

