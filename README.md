## SALICON saliency prediction

### Get the data
[TBD]

### Download the code
git clone https://github.com/cloudinary/dl_salicon_exercise.git

### Build and run the docker
```
cd dl_salicon_exercise
export DATA_DIR=[parent dir of of the SALICON dataset] 
export HOST_WORKDIR=$(pwd)
docker compose -f docker-compose.yml build
docker compose -f docker-compose.yml run salicon
```
### Run training
```
python train.py --config_file train_config.json --data_path /home/data/SALICON --output_path ./output
```

### evaluate crops

```
python evaluate.py --config_file train_config.json --data_path /home/data/SALICON --weight_file ./output/salicon_model.pth --output_path ./evaluate_output
```

### Licence
Released under the MIT license.