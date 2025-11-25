# SeveranceLab

## 의존성
- Python 3.10+
- pandas
- scikit-learn
- seaborn
- matplotlib
- openpyxl (Excel 파일 로딩용)

## 실행 방법
1. 가상환경을 생성하고 필요한 패키지를 설치합니다.
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install pandas scikit-learn seaborn matplotlib openpyxl
   ```
2. 실험 스크립트를 실행하여 교차 검증과 테스트 평가 결과를 저장합니다.
   ```bash
   python experiments/breast_density_experiment.py --data breast_data_250812.xlsx --output-dir results
   ```
3. `results/` 디렉터리 내에 `cv_results.csv/json`, `test_metrics.csv/json`, `threshold_metrics.csv/json` 파일이 생성됩니다.

## 실험 개요
- `experiments/breast_density_experiment.py`는 유방 밀도 등급 3 이상 여부를 이진 타깃으로 만들어 학습합니다.
- 모든 난수 관련 연산에는 `random_state=42`를 사용해 재현성을 유지합니다.
- 교차 검증(`GridSearchCV`) 결과와 테스트 지표, 임계값별 지표를 모두 CSV와 JSON으로 내보냅니다.
