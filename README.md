# Information-Theoretic Bias Assessment Of Learned Representations Of Pretrained Face Recognition
Code for the FG2021 paper:

[Information-Theoretic Bias Assessment Of Learned Representations Of Pretrained Face Recognition](https://arxiv.org/abs/2111.04673)

Jiazhi Li, Wael Abd-Almageed

```
@misc{li2021informationtheoretic,
      title={Information-Theoretic Bias Assessment Of Learned Representations Of Pretrained Face Recognition}, 
      author={Jiazhi Li and Wael Abd-Almageed},
      year={2021},
      eprint={2111.04673},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Requirements
```
pip install -r requirements.txt
```

## Experiments
To conduct experiments, run

```
bash run_colored_model.sh
```

After running, the convergence lines shown as `Fig.3(a)` in the paper will be saved in the current directory as `colored_mnist_convergence.png`. The results with each different color standard deviation will be saved in `./result/colored_model/reproduce/`.



