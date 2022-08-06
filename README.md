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
$ python -m ipykernel install --user --name env_hello --display-name env_hello
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
-rw-rw-r-- 1 jkfirst jkfirst 435656113  8월  2 13:10 squad_model.bin
```

참고 문헌
------------------------
[//]: # (https://asouqi.github.io/bibtex-converter/)

#### 1. chapter1
- Jurafsky, D., & Martin, J. H. (2009). Speech and language processing (2. ed., [Pearson International Edition], p. 1024 S.). Prentice Hall, Pearson Education International. http://aleph.bib.uni-mannheim.de/F/?func=find-b&request=285413791&find_code=020&adjacent=N&local_base=MAN01PUBLIC&x=0&y=0

- Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). Efficient Estimation of Word Representations in Vector Space. CoRR, abs/1301.3781. http://dblp.uni-trier.de/db/journals/corr/corr1301.html#abs-1301-3781

- Sak, H., Senior, A. W., & Beaufays, F. (2014). Long Short-Term Memory Based Recurrent Neural Network Architectures for Large Vocabulary Speech Recognition. CoRR, abs/1402.1128. http://arxiv.org/abs/1402.1128

- Chung, J., Çaglar Gülçehre, Cho, K., & Bengio, Y. (2014). Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling. CoRR, abs/1412.3555. http://arxiv.org/abs/1412.3555

#### 2. chapter2
- Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., & Polosukhin, I. (2017). Attention Is All You Need. CoRR, abs/1706.03762. http://arxiv.org/abs/1706.03762

- Lamb, A., Goyal, A., Zhang, Y., Zhang, S., Courville, A., & Bengio, Y. (2016). Professor Forcing: A New Algorithm for Training Recurrent Networks. arXiv. https://doi.org/10.48550/ARXIV.1610.09038

- Papineni, K., Roukos, S., Ward, T., & Zhu, W.-J. (2002). Bleu: a Method for Automatic Evaluation of Machine Translation. Proceedings of the 40th Annual Meeting of the Association for Computational Linguistics, 311–318. https://doi.org/10.3115/1073083.1073135

#### 3. chapter3
- Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., & Polosukhin, I. (2017). Attention Is All You Need. CoRR, abs/1706.03762. http://arxiv.org/abs/1706.03762

- Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2018). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. CoRR, abs/1810.04805. http://arxiv.org/abs/1810.04805

#### 4. chapter4
- Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2018). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. CoRR, abs/1810.04805. http://arxiv.org/abs/1810.04805
- Lan, Z., Chen, M., Goodman, S., Gimpel, K., Sharma, P., & Soricut, R. (2019). ALBERT: A Lite BERT for Self-supervised Learning of Language Representations. CoRR, abs/1909.11942. http://arxiv.org/abs/1909.11942

- Lan, Z., Chen, M., Goodman, S., Gimpel, K., Sharma, P., & Soricut, R. (2019). ALBERT: A Lite BERT for Self-supervised Learning of Language Representations. CoRR, abs/1909.11942. http://arxiv.org/abs/1909.11942

- Radford, A., Narasimhan, K., Salimans, T., & Sutskever, I. (2018). Improving language understanding by generative pre-training.

- Liu, Y., Ott, M., Goyal, N., Du, J., Joshi, M., Chen, D., Levy, O., Lewis, M., Zettlemoyer, L., & Stoyanov, V. (2019). RoBERTa: A Robustly Optimized BERT Pretraining Approach. arXiv. https://doi.org/10.48550/ARXIV.1907.11692

- Clark, K., Luong, M.-T., Le, Q. V., & Manning, C. D. (2020). ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators. arXiv. https://doi.org/10.48550/ARXIV.2003.10555

- Hinton, G., Vinyals, O., & Dean, J. (2015). Distilling the Knowledge in a Neural Network. arXiv. https://doi.org/10.48550/ARXIV.1503.02531

- Zaheer, M., Guruganesh, G., Dubey, A., Ainslie, J., Alberti, C., Ontañón, S., Pham, P., Ravula, A., Wang, Q., Yang, L., & Ahmed, A. (2020). Big Bird: Transformers for Longer Sequences. CoRR, abs/2007.14062. https://arxiv.org/abs/2007.14062

- Kitaev, N., Kaiser, L., & Levskaya, A. (2020). Reformer: The Efficient Transformer. CoRR, abs/2001.04451. https://arxiv.org/abs/2001.04451

- Gomez, A. N., Ren, M., Urtasun, R., & Grosse, R. B. (2017). The Reversible Residual Network: Backpropagation Without Storing Activations. CoRR, abs/1707.04585. http://arxiv.org/abs/1707.04585

- Wang, A., Singh, A., Michael, J., Hill, F., Levy, O., & Bowman, S. R. (2018). GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding. http://arxiv.org/abs/1804.07461

#### 5. chapter5
- Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., & Sutskever, I. (2018). Language Models are Unsupervised Multitask Learners. https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf

- McAuley, J., & Leskovec, J. (2013). Hidden Factors and Hidden Topics: Understanding Rating Dimensions with Review Text. Proceedings of the 7th ACM Conference on Recommender Systems, 165–172. https://doi.org/10.1145/2507157.2507163

- Nichol, A., Achiam, J., & Schulman, J. (2018). On First-Order Meta-Learning Algorithms. CoRR, abs/1803.02999. http://arxiv.org/abs/1803.02999

- Brown, T. B., Mann, B., Ryder, N., Subbiah, M., Kaplan, J., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A., Agarwal, S., Herbert-Voss, A., Krueger, G., Henighan, T., Child, R., Ramesh, A., Ziegler, D. M., Wu, J., Winter, C., … Amodei, D. (2020). Language Models are Few-Shot Learners. CoRR, abs/2005.14165. https://arxiv.org/abs/2005.14165

#### 6. 부록
- Jacob, B., Kligys, S., Chen, B., Zhu, M., Tang, M., Howard, A. G., Adam, H., & Kalenichenko, D. (2017). Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference. CoRR, abs/1712.05877. http://arxiv.org/abs/1712.05877



