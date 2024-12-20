# project
music style transfer K-pop to carol


# 음악 스타일 변환 프로젝트

## **프로젝트 개요**
이 프로젝트는 딥러닝을 활용하여 음악 스타일 변환 모델을 구현하는 것을 목표로 합니다. K-pop 음악을 입력으로 받아 Carol 스타일의 음악으로 변환, 즉 리메이크 작업을 수행합니다. 

- 오디오 데이터 전처리.
- 음악 스타일 변환을 위한 커스텀 신경망 모델.
- 효율적인 계산을 위한 혼합 정밀도 학습.
- 대용량 데이터를 처리하기 위한 CUDA 메모리 관리.
- 위치 인코딩(Positional Encoding) 및 Transformer 기반 아키텍처.

## 수행 환경 및 사용 라이브러리
   - Python 3.10+
   - PyTorch
   - torchaudio
   - librosa
   - pydub


## **프로젝트 구조**:
1. `data.py`
- 데이터 로딩 및 전처리 담당
- MultiMusicDataseet class : 여러 데이터셋을 통합 관리 하기 위함
- merge_kpop_with_carol : 여러 파일로 이루어진 k-pop 음악 데이터와 Carol 음악 데이터를 병합하여 .pt 파일로 저장, 각 음악 데이터들을 최대 30초 길이로 잘라 저장
  

2. `models.py`
- 주요 모델 정의
- MusicTransformer class
   : 오디오 데이터를 입력받아 스타일 변환을 수행하는 메인 모델
   Conv1d 레이어를 통해 오디오 데이터의 주요 특dlgn 징을 추출하고 positional encoding이후 transformer 모델 적용. 이후 변환된 feature를 다시 오디오 데이터로 변환하기 위한 fc layer
- Positional encoding
- NeuralVocoder
    : MusicTransformer의 출력데이터를 실제 오디오 데이터로 디코딩
3. `train.py`
- 학습 루프 구현
4. `eval.py`
- 테스트 데이터로 모델 평가.
5. `main.py`
- 전체 workflow 통합 및 실행. 데이터 전처리, 모델 학습, 평가 등을 포함한 주요 작업을 실행

## **Work Flow**

### **1. 데이터 전처리**
- 입력 데이터: K-pop 오디오 파일 6개와 Carol 오디오 파일 1개.
- 전처리 단계:
  1. `pydub` 및 `librosa`를 사용하여 오디오 파일을 `numpy` 배열로 변환.
  2. 대상 샘플링 속도를 16kHz로 리샘플링.
  3. 오디오 길이를 최대 30초로 자름
  4. 전처리된 데이터를 `.pt` 파일로 저장.

### **2. 모델 아키텍처**
#### **MusicTransformer**
- **feature extraction**:
  - 초기 오디오 특징 추출을 위한 두 개의 `Conv1d` 레이어.
- **Positional Encoding**:
  - Transformer와 호환되는 오디오 임베딩에 위치 정보를 추가.
- **Transformer Encoder**:
  - 6층 Transformer 인코더와 8개의 어텐션 헤드.
- **출력 레이어**:
  - 최종 오디오 스타일 변환을 위한 완전 연결 레이어.

#### **Neural Vocoder**
- 잠재 특징(latent feature)을 다시 오디오 파형으로 디코딩.

### **3. 학습**
- **최적화 알고리즘**: Adam
- **손실 함수**: 평균 제곱 오차(MSE)
- **혼합 정밀도 학습**:
  - `torch.cuda.amp`를 사용하여 메모리 사용량 감소 및 계산 속도 증가.

### **4. 평가**
- 변환된 오디오와 Carol 스타일 오디오를 손실 지표로 비교.
- 수동 검사(manual inspection)를 위한 변환된 오디오 출력.

## **문제 및 해결 방법**

### **1. 대용량 데이터 크기**
- **문제**: 입력 오디오 파일의 크기가 메모리 용량을 초과.
- **해결 방법**:
  - 리샘플링과 자르기 작업을 청크 단위로 처리.
  - 배치 크기 감소.

### **2. 위치 인코딩 제한**
- **문제**: 시퀀스 길이가 위치 인코딩 제한을 초과.
- **해결 방법**:
  - 입력 크기에 따라 위치 인코딩을 동적으로 생성.
  - 입력 시퀀스를 제한된 길이로 자름.

### **3. CUDA 메모리 문제**
- **문제**: 학습 중 메모리 부족 오류 발생.
- **해결 방법**:
  - `torch.cuda.amp`를 사용한 혼합 정밀도 학습 도입.
  - `torch.cuda.empty_cache()`를 사용하여 사용하지 않는 메모리 정리.

## **실행 방법**

1. **데이터 전처리 및 모델학습**:

   ```bash
   python main.py
   ```

2. **모델 평가**: 
   ```bash
   python eval.py
   ```

## **주요 결과**
- K-pop 오디오를 성공적으로 Carol 스타일로 변환.

