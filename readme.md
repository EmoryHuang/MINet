# Joint Modeling of Multimodal Information based on Dynamic and Static Knowledge Graphs for Next POI Recommendation

## Requirements

```
python==3.9.16
pytorch==1.12.1
pandas==1.5.3
numpy==1.24.3
tqdm==4.65.0
matplotlib==3.7.1
pygeohash==1.2.0
torch_geometric==2.3.1
pyarrow==11.0.0
transformers==4.29.2
```

You can install the requirements using `pip install -r requirements.txt`.

## Datasets

Put [MINet_dataset.zip](https://drive.google.com/file/d/1yVQsUPHqW59Bz6CBmLGMJuknVyW_zqLU/view?usp=sharing) into `./Datasets/`.

> And the original data can be found [here](https://www.yelp.com/dataset)
> 
> If you are using the original dataset, follow these steps for data processing:
> 
> 1. Generate Dataset: use `python preprocess.py`
> 2. ABSA Model Pretraining: use `python pretraining.py --gpu=0`
> 3. Review test: use `python pretraining.py --mode=test --gpu=0`

## Pretrained Model

The checkpoint file can be found [here](https://drive.google.com/file/d/1A_JSyWT___Z53LP8fdbSYoG8vwOTgtIw/view?usp=sharing) and put them into `./Model/`.

## Training

```bash
python main.py --mode=train --dataset=yelp --gpu=0
```

## Evaluation

```bash
python main.py --mode=test --dataset=yelp --model_path='./Model/model_yelp.pkl' --gpu=0
```