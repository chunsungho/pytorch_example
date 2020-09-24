### 이 코드는 resnet-20, 32, 44, 56과 plainNet-20,32,44,56을 직접 학습하여 train err, test err각각을 list형으로 저장하는 예제입니다.

# main.py
## Hyper-parameter
### RandomCropping after Padding 4 each side
### weight initialization = torch.nn.init.kaiming_uniform_()
### weight decay = 0.0001
### momoentum = 0.9
### use bn, no dropout
### learning rate : 0.1(~32k) -> 0.01(~48k) -> 0.001(~64k)
### mini-batch size : 128

# read_list
main.py 코드를 이용해 학습을 완료하면 지정한 경로에 train err, test err가 list로 저장됩니다. read_list.py는 이 list를 읽어서 matplotlib 라이브러리를 이용해 그래프를 그리는 예제입니다.
