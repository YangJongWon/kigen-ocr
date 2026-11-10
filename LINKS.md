# kigenocr 참고 링크

일본 상품의 賞味期限·消費期限 OCR에 쓸 데이터셋과 참고 자료 모음 (2026-11-09 조사)

> 전제: **일본어 유통기한만 모은 공개 데이터셋은 찾지 못함.** 한자 형식(年月日, 令和, 旬, 時分)은 합성 데이터(`gen.py`)로 학습하고, 아래 데이터는 주로 **실사진 평가**용으로 쓴다.

## 1. 데이터셋

### ExpDate (Seker & Ahn, 2022) — 1순위
- 식품·음료·의약품 실사진 **1,767장** (Products-Real), 합성 세트와 날짜 영역만 잘라 낸 세트(Date-Real / Date-Synth)도 있음
- 날짜는 주로 숫자·영어 표기 → 숫자, 점·슬래시 구분자, 도트 인쇄 평가에 적합. 한자 형식은 거의 없음
- **라이선스 미확정**: 미러마다 다르게 적혀 있음(AFL-3.0 / Apache-2.0). 원 저작자 페이지에서 확인할 것. 상업용이면 학습에는 넣지 말고 평가에만 사용
- 링크
  - 프로젝트 페이지: https://felizang.github.io/expdate
  - 원 홈페이지(로더 스크립트에 적힌 주소): https://acseker.github.io/ExpDateWebsite
  - Hugging Face 미러 (Products-Real / Synth): https://huggingface.co/datasets/dimun/ExpirationDate
  - Hugging Face 미러 (Date-Real / Synth): https://huggingface.co/datasets/dimun/dates

### Roboflow Universe — 날짜 위치 검출용 (참고)
- 대부분 날짜 영역을 박스로 표시한 검출용이고, 문자 라벨은 거의 없음. 규모가 작고 라이선스는 프로젝트마다 다름
- https://universe.roboflow.com/hochschule-fuer-angewandte-wissenschaften-hamburg/expiry-date-detection
- https://universe.roboflow.com/grid2/expiry-date-ocr
- https://universe.roboflow.com/boxmilk/exp_date-q5hqo
- https://universe.roboflow.com/college-37gbk/expiry-date-detection-wowdr

### Open Food Facts — 실사진 후보 수집용
- 포장 사진 수백만 장(일본 상품 포함). 유통기한 라벨은 없어서 그대로는 학습에 못 씀. 데이터 라이선스는 ODbL, 사진은 CC BY-SA 계열이므로 사용 전 확인
- https://blog.openfoodfacts.org/?p=4209

## 2. 논문·참고 자료

| 자료 | 내용 | 링크 |
|---|---|---|
| Florea & Rebedea (2020) | 도트 인쇄 유통기한 인식. 검증 세트를 GitHub에 공개했다고 함(미확인) | https://rochi.utcluj.ro/ijusi/articles/IJUSI-13-1-Florea.pdf |
| Lotfy & Soliman (2024) | 아랍어 도트 인쇄 유통기한. 실데이터 부족으로 합성 데이터 사용 | https://jesit.springeropen.com/articles/10.1186/s43067-024-00136-2 |
| Springer (2025) | 아랍어 도트 인쇄. 13가지 날짜 형식 | https://link.springer.com/article/10.1186/s43067-025-00210-3 |
| IEICE (2019) 竹内 | 시각장애인용 賞味期限 읽기. 기존 OCR의 날짜 인식률 40% | https://ken.ieice.org/ken/paper/20190810s1PX/ |
| IEICE (2013) | 카메라로 찍은 도트 매트릭스 OCR | https://ken.ieice.org/ken/paper/20131213dB70/eng/ |

## 3. 다음에 할 일
- [ ] ExpDate 원 페이지에서 라이선스 확인
- [ ] Products-Real로 현재 모델 평가 (숫자 형식만 따로 집계)
- [ ] 일본 슈퍼 상품 실사진 20~30장 직접 수집 → 평가용 세트
- [ ] Florea GitHub 검증 세트 위치 확인
