# COSE474-Final-project
주제 : SST2 dataset과 IMDB dataset을 활용한 binary sentiment analysis

코드를 실행시키기 위해 필요한 패키지와 라이브러리 버전은 requirement.txt를 참조하시면 됩니다.
저장된 csv 파일은 IMDB dataset을 8:2로 구분하여 train dataset과 test dataset으로 분류한 파일입니다. 
SST2 dataset은 'datasets'을 이용하여 load합니다.

코드 실행 과정은 다음과 같습니다.

**1. SST-2 dataset으로만 fine-tuning할 경우**
   
   SST2_finetuning.py 파일을 실행하면 됩니다.

**2. IMDB dataset으로만 fine-tuning할 경우**
   
   IMDB_finetuning.py 파일을 실행하면 됩니다.

**3. SST-2 dataset으로 1차 fine-tuning, 이 후 IMDB dataset으로 2차 fine-tuning하는 경우**
   
   SST2_finetuning.py 파일을 실행한 후, IMDB_finetuning.py에서 model.load_state_dict을 활용하여 SST2로 fine-tuning했던 모델의 pt 파일을 불러오시면 됩니다.




3가지 방법으로 훈련된 모델의 성능을 평가하기 위해 test.py 파일을 실행하면 됩니다.
테스트 결과는 result_figure 폴더에 자동으로 저장됩니다.
