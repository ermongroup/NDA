# NDA

To train with Jigsaw NDA for unconditional Cifar-10, run the following - 
```
bash train_C10.sh
```
To evaluate the trained model, run - 
```
bash eval_C10.sh jigsaw_C10
```
For conditional Cifar-10, run the following - 
```
bash train_C10_cond.sh
```

To evaluate the trained model, run - 
```
bash eval_C10_cond.sh jigsaw_C10_cond
```

## Evaluating pre-trained model

To evaluate pretrained model for unconditional Cifar-10, run the following - 
```
bash eval_C10.sh jigsaw_seed2_C10_alpha_0.25_beta_0.75
```
For conditional Cifar-10, run the following - 
```
bash eval_C10_cond.sh jigsaw_C10_conditional_seed2_alpha_0.25_beta_0.75
```
## Using other NDA
Lines 242-246 in train_fns_aug.py contain other NDA augmentations, uncomment the corresponding line to use that NDA. Change the experiment_name argument in train_C10.sh or train_C10_cond.sh to generate a seperate model for that NDA

