# AGI-Project

### dataset
```
MIMIC-CXR
CheXpert
ChestImaGenome
```

## Install
- 필요 library 전체 설치
```
pip install -r requirements.txt
```
- Image-to-Text, Text-to-KG, Image-to-KG 각각 개별로 설치
```
cd [Image-to-Text/Text-to-KG/Image-to-KG]
pip install -r requirements.txt
```

## Image to Text
- ResNet101과 Memory 모듈을 결합하여, 이미지의 시각적 특징과 텍스트 패턴을 정교하게 추출
- Transformer 디코더를 통해 문맥을 구성하고, Beam Search 최적화 과정을 거쳐 반복 오류를 억제한 텍스트를 생성
### dataset
- [여기](https://physionet.org/content/mimic-cxr/2.0.0/) `MIMIC-CXR` 에서 데이터 세트를 다운로드한 다음 파일을 `data/mimic-cxr` 경로에 넣어주세요
  - id : 각 샘플의 고유 식별자
  - image_path : 이미지 파일 경로를 담은 리스트
  - report : 이미지에 대한 설명 또는 라벨링 텍스트
  - split : 데이터셋 구분 (`train`, `val`, `test`)
  ```
  [
    {
        "id": "unique_sample_id",
        "image_path": [
        "/path/to/your/dataset/images/image_filename.jpg"
        ],
        "report": "Full text description or medical report corresponding to the image.",
        "split": "train"
    },
    {
        "id": "unique_sample_id_2",
        "image_path": [
        "/path/to/your/dataset/images/image_filename_2.jpg"
        ],
        "report": "Another sample report text.",
        "split": "test"
    }
  ]


## Text to Knowledge Graph
- 방사선 report에서 의학적 지식을 담은 triplet을 추출하는 모듈
- RadGraph를 활용하여 방사선 report에서 triplet을 추출하고, Neo4j를 통해 시각화 함  
[지식그래프 시각화 사진]


## Image to Knowledge Graph
- MIMIC-CXR 리포트 텍스트를 분석하여 해부학적 영역을 필터링하고, 이미지 위에 Bounding Box로 시각화
- X-ray 이미지를 중심으로 해부학적 객체와 병변 속성의 관계를 구조화하여 신그래프 형태로 생성
### dataset
- [여기](https://physionet.org/content/chest-imagenome/1.0.0/) `ChestImaGenome` 에서 데이터 세트를 다운로드한 다음 파일을 `data/chestimagenome` 경로에 넣어주세요
[11월 7일 세미나 ppt에 들어간 mmkg 이미지 넣기]

## Reference
- R2GenCMN: Cross-model memory networks for radiology generation, arXiv 2022, Chen, Zhihong., et al.
- RadGraph: Extracting Clinical Entities and Relations from Radiology Reports, NeurlIPS 2021, Jean-Benoit Delbrouck., et al.
- Chest ImaGenome Dataset for Clinical Reasoning, NeurlIPS 2021, Wu, Joy T., et al.
- SGRRG: Leveraging radiology scene graphs for improved and abnormality-aware radiology report generation, CMIG 2025, WANG, Jun., et al.
