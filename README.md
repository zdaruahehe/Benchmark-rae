# Benchmark-rae
# Benchmarking Deterministic Autoencoders for Generative Modeling: Image generation

#### Written by Yuhan Xie

## Setting Up
The code is under the following setup:
* Using this version of Anaconda: https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh, 
* and whole environment setup is in environment.yaml called rae
* to setup environment, please use "conda env create -f environment.yaml" and enter "conda activate rae" in anaconda command prompt.

## Folder structure
The submitted material "myproject.zip" contains One folder:
* RAE
* 
This Github repository displays the specific files directory under the RAE folder

#### To navigate around the data folder structure
```
RAE
 |____ data
        |___ cifar-10-batches-py
        |
        |___ FashionMNIST     
        |            
        |___ mnist
        |
        |____my_dataset.py
		  

```
Due to file size limitation, all other data are found in the google drive link: https://drive.google.com/drive/folders/1J2onGDemUzY0jZ3kD4JGXH6voUvAip4N?usp=sharing

#### To navigate around the trained result folder structure, folder "train_result" is for generated images, and folder "trained_models" is for trained models checkpoint.
```
RAE
 |____ train_result 
            |___ generated_images_for_model1
            |      |___reconstructed
            |      |___samples
            |
            |___ generated_images_for_model2
            |      |___reconstructed
            |      |___samples
            |___ ...(generated images for diffenret models)
            |   
            |___ GMM_images
		   |___GMM_images_for_model1
                   |___...(GMM samples for different trained models)

```

```
RAE
 |____ trained_models 
            |___ trained_model1
            |
            |___ trained_model2
            |
            |___ ...

```

Due to file size limitation, all training results are found in the google drive link: https://drive.google.com/drive/folders/1J2onGDemUzY0jZ3kD4JGXH6voUvAip4N?usp=sharing

#### Training 
* All training scripts are in "RAE" folder,
* Model training scripts are named starting with the prefix "train" and network name followed by the RAE, VAE or WAE
* Note that folders "network", "opts" and "utils" are packages depended by training scripts above. Loss functions of different networks are in diffenrent sub-folders named with model name under folder "network".
* which is

```
RAE
 |____ network 
          |___ rae
                |___ loss_function
                |___ ...(model structures)
          |
          |___ SN_layers
                |___ ...(model structures)
          |
          |___ vae
                |___ loss_function
                |___ ...(model structures)
          |
          |___ wae
                |___ loss_function
                |___ ...(model structures)

```

####  Evaluation
* All evaluation scripts are in "RAE" folder,
* Model Evaluation for images generation are found in calculate_fid.py,
* To run evaluation script calculate_FID.py, please specify two pathes of destination folders which both contain images, and a parameter "--gpu": none for cpu, a number for gpu such as "--gpu 0"
* that is enter "python calculate_FID.py path1 path2 --gpu 0"
* Due to file size limitation, some of the files like inception_v3_google-0cc3c7bd.pth are uploaded on the google drive: https://drive.google.com/drive/folders/1J2onGDemUzY0jZ3kD4JGXH6voUvAip4N?usp=sharing

