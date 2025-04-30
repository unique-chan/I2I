# I2I
Image-to-image translation

* https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix

~~~shell
conda create -n i2i python=3.8 -y
conda activate i2i
conda install pytorch==2.1.2 torchvision==0.16.2 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install dominate==2.9.1
pip install visdom==0.2.4
pip install wandb==0.19.10
~~~

훈련 연습 (1)
- 코드에서 제공하는 horse2zebra 데이터셋 활용
  - 데이터셋은 사전에 다운로드할 것: 
      ~~~shell
      cd pytorch-CycleGAN-and-pix2pix/
      bash ./datasets/download_cyclegan_dataset.sh horse2zebra
      ~~~

- Learning curve를 실시간으로 시각적으로 살펴보려면, 아래 코드 실행
~~~shell
visdom -port 8097
~~~

- 훈련 예시 (5에폭)
~~~shell
cd pytorch-CycleGAN-and-pix2pix/
python train.py  --dataroot ./datasets/horse2zebra --name YECHANhorse2zebra --model cycle_gan \
                 --display_id -1 --n_epochs 5 --n_epochs_decay 100
~~~


훈련 연습 (2)
- AMOD (Ours) 및 DOTA 데이터 활용

- 훈련 예시 (5에폭) (AMOD 시야각 `10도`만 활용!)
~~~shell
cd pytorch-CycleGAN-and-pix2pix/
python train.py  --dataroot "" \
                 --datarootA "/media/yechani9/KYC_AMD/AMOD_V1_FINAL_OPTICAL" \
                 --datarootB "/media/yechani9/KYC_AMD/DOTA-v1.0-for-mmrotate" \
                 --filterA "*/10/*.png" \
                 --preprocess "random_scale_width_and_crop" --load_size 542 --crop_size 512 \
                 --name "amod2dota10" --model cycle_gan \
                 --display_id -1 --n_epochs 100 --n_epochs_decay 100
~~~

