# SeveranceLab

로지스틱 회귀 기반의 그리드 서치를 재현할 수 있도록 예시 스크립트와 실행 방법을 정리했습니다.

## 환경 준비

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 데이터셋

- CSV 또는 Excel 파일 형태를 지원합니다(`.csv`, `.xlsx`).
- 기본 타깃 컬럼 이름은 `target`이므로, 다를 경우 `--target-column` 옵션으로 지정합니다.
- 모든 특징 컬럼은 숫자형이어야 합니다. 범주형이 포함되어 있다면 원-핫 인코딩 등으로 미리 변환해주세요.

## 그리드 서치 실행

다음 예시는 제공된 `breast_data_250812.xlsx` 파일을 사용해 5-겹 ROC AUC 기준의 하이퍼파라미터 탐색을 수행합니다.

```bash
python grid_search.py breast_data_250812.xlsx --target-column target --top-n 5
```

주요 탐색 파라미터는 다음과 같습니다.

- `C`: `[0.01, 0.1, 1, 10, 100]`
- `penalty`: `['l2', 'elasticnet']` (`elasticnet`인 경우 `l1_ratio`: `[0.1, 0.5, 0.9]`)
- `solver`: `['saga', 'lbfgs']` (`elasticnet`은 `saga`만 사용)
- `class_weight`: `[None, "balanced"]`

`GridSearchCV(pipeline, param_grid, scoring="roc_auc", cv=5, n_jobs=-1)` 구성으로 실행되며, 결과로는 베스트 파라미터, 상위 조합, 테스트 세트 ROC AUC 및 분류 리포트를 출력합니다. 필요에 따라 `--top-n`으로 표시할 상위 조합 개수를 조정하고, 출력된 `cv_results_`를 참고해 `C`나 `l1_ratio` 범위를 좁히거나 확장할 수 있습니다.
