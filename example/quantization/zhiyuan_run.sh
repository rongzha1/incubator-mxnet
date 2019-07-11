#!/bin/bash


function Run_imagenet_gen_qsym_mkldnn () {
    # export MXNET_SUBGRAPH_BACKEND=MKLDNN
    python imagenet_gen_qsym_mkldnn.py --model=resnet50_v1 --num-calib-batches=5 --calib-mode=naive
    python imagenet_gen_qsym_mkldnn.py --model=resnet101_v1 --num-calib-batches=5 --calib-mode=naive
    python imagenet_gen_qsym_mkldnn.py --model=squeezenet1.0 --num-calib-batches=5 --calib-mode=naive
    python imagenet_gen_qsym_mkldnn.py --model=mobilenet1.0 --num-calib-batches=5 --calib-mode=naive
    python imagenet_gen_qsym_mkldnn.py --model=inceptionv3 --image-shape=3,299,299 --num-calib-batches=5 --calib-mode=naive
    python imagenet_gen_qsym_mkldnn.py --model=imagenet1k-resnet-152 --num-calib-batches=5 --calib-mode=naive
    python imagenet_gen_qsym_mkldnn.py --model=imagenet1k-inception-bn --num-calib-batches=5 --calib-mode=naive
}

function RealFP32 () {
    python imagenet_inference.py --symbol-file=./model/resnet50_v1-symbol.json --param-file=./model/resnet50_v1-0000.params --rgb-mean=123.68,116.779,103.939 --rgb-std=58.393,57.12,57.375 --num-skipped-batches=50 --batch-size=64 --num-inference-batches=500 --dataset=./data/val_256_q90.rec --ctx=cpu --data-nthreads=1
    python imagenet_inference.py --symbol-file=./model/resnet101_v1-symbol.json --param-file=./model/resnet101_v1-0000.params --rgb-mean=123.68,116.779,103.939 --rgb-std=58.393,57.12,57.375 --num-skipped-batches=50 --batch-size=64 --num-inference-batches=500 --dataset=./data/val_256_q90.rec --ctx=cpu --data-nthreads=1
    python imagenet_inference.py --symbol-file=./model/squeezenet1.0-symbol.json --param-file=./model/squeezenet1.0-0000.params --rgb-mean=123.68,116.779,103.939 --rgb-std=58.393,57.12,57.375 --num-skipped-batches=50 --batch-size=64 --num-inference-batches=500 --dataset=./data/val_256_q90.rec --ctx=cpu --data-nthreads=1
    python imagenet_inference.py --symbol-file=./model/mobilenet1.0-symbol.json --param-file=./model/mobilenet1.0-0000.params --rgb-mean=123.68,116.779,103.939 --rgb-std=58.393,57.12,57.375 --num-skipped-batches=50 --batch-size=64 --num-inference-batches=500 --dataset=./data/val_256_q90.rec --ctx=cpu --data-nthreads=1
    python imagenet_inference.py --symbol-file=./model/inceptionv3-symbol.json --param-file=./model/inceptionv3-0000.params --image-shape=3,299,299 --rgb-mean=123.68,116.779,103.939 --rgb-std=58.393,57.12,57.375 --num-skipped-batches=50 --batch-size=64 --num-inference-batches=500 --dataset=./data/val_256_q90.rec --ctx=cpu --data-nthreads=1
    python imagenet_inference.py --symbol-file=./model/imagenet1k-resnet-152-symbol.json --param-file=./model/imagenet1k-resnet-152-0000.params --num-skipped-batches=50 --batch-size=64 --num-inference-batches=500 --dataset=./data/val_256_q90.rec --ctx=cpu --data-nthreads=1
    python imagenet_inference.py --symbol-file=./model/imagenet1k-inception-bn-symbol.json --param-file=./model/imagenet1k-inception-bn-0000.params --rgb-mean=123.68,116.779,103.939 --num-skipped-batches=50 --batch-size=64 --num-inference-batches=500 --dataset=./data/val_256_q90.rec --ctx=cpu --data-nthreads=1

}

function RealInt8 () {
    python imagenet_inference.py --symbol-file=./model/resnet50_v1-quantized-5batches-naive-symbol.json --param-file=./model/resnet50_v1-quantized-0000.params --rgb-mean=123.68,116.779,103.939 --rgb-std=58.393,57.12,57.375 --num-skipped-batches=50 --batch-size=64 --num-inference-batches=500 --dataset=./data/val_256_q90.rec --ctx=cpu  --data-nthreads=1
    python imagenet_inference.py --symbol-file=./model/resnet101_v1-quantized-5batches-naive-symbol.json --param-file=./model/resnet101_v1-quantized-0000.params --rgb-mean=123.68,116.779,103.939 --rgb-std=58.393,57.12,57.375 --num-skipped-batches=50 --batch-size=64 --num-inference-batches=500 --dataset=./data/val_256_q90.rec --ctx=cpu  --data-nthreads=1
    python imagenet_inference.py --symbol-file=./model/squeezenet1.0-quantized-5batches-naive-symbol.json --param-file=./model/squeezenet1.0-quantized-0000.params --rgb-mean=123.68,116.779,103.939 --rgb-std=58.393,57.12,57.375 --num-skipped-batches=50 --batch-size=64 --num-inference-batches=500 --dataset=./data/val_256_q90.rec --ctx=cpu  --data-nthreads=1
    python imagenet_inference.py --symbol-file=./model/mobilenet1.0-quantized-5batches-naive-symbol.json --param-file=./model/mobilenet1.0-quantized-0000.params --rgb-mean=123.68,116.779,103.939 --rgb-std=58.393,57.12,57.375 --num-skipped-batches=50 --batch-size=64 --num-inference-batches=500 --dataset=./data/val_256_q90.rec --ctx=cpu  --data-nthreads=1
    python imagenet_inference.py --symbol-file=./model/inceptionv3-quantized-5batches-naive-symbol.json --param-file=./model/inceptionv3-quantized-0000.params --image-shape=3,299,299 --rgb-mean=123.68,116.779,103.939 --rgb-std=58.393,57.12,57.375 --num-skipped-batches=50 --batch-size=64 --num-inference-batches=500 --dataset=./data/val_256_q90.rec --ctx=cpu  --data-nthreads=1
    python imagenet_inference.py --symbol-file=./model/imagenet1k-resnet-152-quantized-5batches-naive-symbol.json --param-file=./model/imagenet1k-resnet-152-quantized-0000.params --num-skipped-batches=50 --batch-size=64 --num-inference-batches=500 --dataset=./data/val_256_q90.rec --ctx=cpu  --data-nthreads=1
    python imagenet_inference.py --symbol-file=./model/imagenet1k-inception-bn-quantized-5batches-naive-symbol.json --param-file=./model/imagenet1k-inception-bn-quantized-0000.params --rgb-mean=123.68,116.779,103.939 --num-skipped-batches=50 --batch-size=64 --num-inference-batches=500 --dataset=./data/val_256_q90.rec --ctx=cpu  --data-nthreads=1

}

function dummyFP32 () {
    python imagenet_inference.py --symbol-file=./model/resnet50_v1-symbol.json --batch-size=64 --num-inference-batches=500 --ctx=cpu --benchmark=True
    python imagenet_inference.py --symbol-file=./model/resnet101_v1-symbol.json --batch-size=64 --num-inference-batches=500 --ctx=cpu --benchmark=True
    python imagenet_inference.py --symbol-file=./model/squeezenet1.0-symbol.json --batch-size=64 --num-inference-batches=500 --ctx=cpu  --benchmark=True
    python imagenet_inference.py --symbol-file=./model/mobilenet1.0-symbol.json --batch-size=64 --num-inference-batches=500 --ctx=cpu  --benchmark=True
    python imagenet_inference.py --symbol-file=./model/inceptionv3-symbol.json --image-shape=3,299,299 --batch-size=64 --num-inference-batches=500 --ctx=cpu  --benchmark=True
    python imagenet_inference.py --symbol-file=./model/imagenet1k-resnet-152-symbol.json --batch-size=64 --num-inference-batches=500 --ctx=cpu --benchmark=True
    python imagenet_inference.py --symbol-file=./model/imagenet1k-inception-bn-symbol.json --batch-size=64 --num-inference-batches=500 --ctx=cpu --benchmark=True

}

function dummyInt8 () {
    python imagenet_inference.py --symbol-file=./model/resnet50_v1-quantized-5batches-naive-symbol.json --batch-size=64 --num-inference-batches=500 --ctx=cpu --benchmark=True
    python imagenet_inference.py --symbol-file=./model/resnet101_v1-quantized-5batches-naive-symbol.json --batch-size=64 --num-inference-batches=500 --ctx=cpu --benchmark=True
    python imagenet_inference.py --symbol-file=./model/squeezenet1.0-quantized-5batches-naive-symbol.json --batch-size=64 --num-inference-batches=500 --ctx=cpu --benchmark=True
    python imagenet_inference.py --symbol-file=./model/mobilenet1.0-quantized-5batches-naive-symbol.json --batch-size=64 --num-inference-batches=500 --ctx=cpu --benchmark=True
    python imagenet_inference.py --symbol-file=./model/inceptionv3-quantized-5batches-naive-symbol.json --image-shape=3,299,299 --batch-size=64 --num-inference-batches=500 --ctx=cpu --benchmark=True
    python imagenet_inference.py --symbol-file=./model/imagenet1k-resnet-152-quantized-5batches-naive-symbol.json --batch-size=64 --num-inference-batches=500 --ctx=cpu --benchmark=True
    python imagenet_inference.py --symbol-file=./model/imagenet1k-inception-bn-quantized-5batches-naive-symbol.json --batch-size=64 --num-inference-batches=500 --ctx=cpu --benchmark=True

}
# USE MKLDNN AS SUBGRAPH BACKEND
export MXNET_SUBGRAPH_BACKEND=MKLDNN
# export MXNET_DISABLE_MKLDNN_FUSE_CONV_RELU=1
export MXNET_DISABLE_MKLDNN_FUSE_CONV_SUM=1
#Run_imagenet_gen_qsym_mkldnn
#RealFP32
RealInt8

