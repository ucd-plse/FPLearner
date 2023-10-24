### How to build the docker image by your own?


docker build -t rainytong/fpl_cuda_11.7 .

docker run -ti --gpus all  --name test rainytong/fpl_cuda_11.7

