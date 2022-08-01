안녕 트랜스포머
------------------------

이 레포지토리는 "안녕 트랜스포머"의 소스 코드 레포지토리입니다.


파이선 패키지 설치
------------------------
1. 가상환경을 설치합니다. (파이선 버전은 3.6 이상을 권고합니다)
```
$ virtualenv -p /usr/bin/python3.8 env_hello
$ source env_hello/bin/activate
```

2. pip을 사용해서 파이선 패키지를 설치합니다.
```
# cpu 환경에서 사용할 경우, chapter4의 BERT 파인튜닝 모델의 학습을 위한 .ipynb 파일 실행이 어려움
$ pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu

# cuda 버전이 10.2일 경우
$ pip install torch torchvision torchaudio

# cuda 버전이 11.3일 경우
$ pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113

# cuda 버전이 11.6일 경우
$ pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116

# torch 이외의 패키지 설치
$ pip install -r requirements.txt
```

3. 주피터 노트북 커널을 추가합니다.
```
$ python -m ipykernel install --user --name env_hello_two --display-name env_hello_two
```

모델 파일
------------------------
이 책에서 사용하는 모델 파일 중에서 크기가 크지 않은 파일은 git 레포지토리를 통해 직접 다운로드 받을 수 있습니다. 큰 모델 파일의 경우에는 아래와 같이 쪼개진 tar.gz 파일을 다시 합쳐서 압축을 푸는 방식으로 얻을 수 있습니다.
```
# research/chapter4/cola_classification
$ cd research/chapter4/cola_classification/models/
$ cat models.tar.gz.parta* > models.tar.gz
$ tar xvfz models.tar.gz
$ ls -al *.bin
-rw-rw-r-- 1 jkfirst jkfirst 438019245  8월  2 02:47 cola_model.bin
-rw-rw-r-- 1 jkfirst jkfirst 438019245  8월  2 02:56 cola_model_no_pretrained.bin

# research/chapter4/squad
$ cd research/chapter4/squad/models
$ cat models.tar.gz.parta* > models.tar.gz
$ tar xvfz models.tar.gz
$ ls -al *.bin
```

참고 문헌
------------------------
1. chapter1
```
```

2. chapter2
```
Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., & Polosukhin, I. (2017). Attention Is All You Need. CoRR, abs/1706.03762. http://arxiv.org/abs/1706.03762
```

3. chapter3
```
```

4. chapter4
```
```

5. chapter5
```
```

6. 부록
```
```
