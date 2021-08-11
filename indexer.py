# -*- coding: utf-8 -*-

import re
import argparse

pat = '^(\d+\.)+'

doc = '''
5.2. BERT
BERT는 2018년 10월에 발표된 언어 모델이며, 논문 제목은 "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"이다. BERT는 레이블이 없는 데이터로부터 레이블을 스스로 생성하여 사전학습을 진행해서 만들어진 Pretrained 모델이다. 이 모델을 레이블이 있는 데이터셋으로 Fine-Tuning해서 11개의 자연어 처리 테스크에서 SoTA를 달성했다.

5.2.1. 사전학습 이해하기
BERT는 사전 학습된 언어모델이다. BERT의 구조는 4장에서 소개한 트랜스포머의 인코더 구조를 띄고 있는데 이 구조에 데이터를 넣어서 미리 학습시킨 것이 사전학습된 BERT이고, 이 사전학습된 BERT는 이미 Google이나 HuggingFace, SKT 등등의 큰 기업 또는 오픈소스 프로젝트에서 많은 컴퓨팅 리소스를 이용해서 학습해뒀고, 학습된 모델을 다운로드 받아서 활용할 수 있다. 

그러면 어떻게 BERT를 사전학습했을까? 직접 학습을 해보기는 힘들어도 원리를 이해해볼 수는 있다. BERT의 사전학습은 언어에 대한 사전학습이다. 따라서 언어를 학습할 때 어떤 식으로 학습하는지를 먼저 생각해보자. Masked Language Model(MLM)과 Next Sentence Prediction(NSP)를 통해 학습을 한다. BERT는 MLM과 NSP의 Loss를 낮추는 방식으로 학습된다. 그러면 MLM과 NSP가 각각 어떤 학습을 하는지 다음 절에서 자세하게 알아보자.

5.2.2. BERT의 입력 살펴보기
BERT에 대해서 자세하게 살펴보기 전에 우선 BERT의 입력 데이터가 어떤 형태를 띄는지 살펴보자. 4장 트랜스포머는 기계 번역을 학습한 모델이기 떄문에 입력 데이터가 번역할 문장(source)과 번역된 문장(target)으로 이뤄져 있었고, 이 문장들은 토크나이져를 통해서 토큰화되어 입력됐다. BERT도 트랜스포머처럼 토크나이저를 통해 트큰화된 문장이 입력으로 사용된다. 아래의 예시를 보자.
<코드 시작>
'''

def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('txt_filename', type=str, help='text filename')
    args = parser.parse_args()

    with open(args.txt_filename, 'r') as f:
        for oneline in f:
            oneline = oneline.rstrip()
            matched = re.match(pat, oneline)
            if matched:
                print(oneline)


if __name__ == '__main__':
    main()
