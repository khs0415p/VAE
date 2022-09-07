# Variational autoencoders

## Data

<br>

- MNIST

---

## Models

<br>

- Variational autoencoders

---

## How to use

<br>

`config.json`파일로 컨트롤이 가능합니다.

`denoising`을 1과 0의 값으로 `Denosing`설정이 가능합니다.



- config 설정 후 학습
```
python3 main.py --mode train --gpu
```

<br>

- 학습 완료 후 테스트

```
python3 main.py --mode test --gpu --name {model name}
```

<br>

- 각 모델의 개념은 [여기](https://velog.io/@khs0415p/VAE-Variational-Autoencoders)에 정리 되어있습니다.

---

## results

<br>

- VAE

<img src="./imgs/VAE_result.png"  width="300" height="700"/> <br><br>


- VAE manifold

<img src="./imgs/VAE_2D.png"  width="500" height="500"/> <br><br>


- gif

<img src="./imgs/res.gif"  width="300" height="300"/> <br><br>