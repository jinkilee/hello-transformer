안녕 트랜스포머
------------------------

이 레포지토리는 "안녕 트랜스포머"의 소스 코드 레포지토리입니다.


파이선 패키지 설치
------------------------
1. 가상환경을 설치합니다.
```
$ virtualenv -p /usr/bin/python3.8 env_hello
$ source env_hello/bin/activate
```

2. pip을 사용해서 파이선 패키지를 설치합니다.
```
$ pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu
$ pip install torch torchvision torchaudio
$ pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
$ pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
$ pip install -r requirements.txt
```

3. 주피터 노트북 커널을 추가한다.
```
$ python -m ipykernel install --user --name env_hello_two --display-name env_hello_two
```




