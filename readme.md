# install
* picas-server와 같은 레벨에 모듈을 위치시킵니다.  
    
# config
* data: data path
* mask: mask path if loaded from file system
* val-data: validate data path
* batch_size: batch size
* lr: learning rate
* max-iter: training max iteration
* input-size: image input size 
* checkpoint: checkpoint save point path
* iter-log: how many iterations for logging (0: never)
* iter-save: how many iterations for saving (0: never)
* iter-sample: how many iterations for sampling (0: never)
* iter-lr: how many iterations for learning rate decay * 0.5 (0: never)
* dataset: dataset for trainig ( lsun | folder)
* tensor-log: tensorboard logdir path

# training
    python main.py <options>