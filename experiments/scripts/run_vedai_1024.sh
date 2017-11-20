/bash

LOG="../logs/vedai_1024.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

cd ../../

./tools/train_net.py --gpu 0 --weights data/imagenet_models/VGG16.v2.caffemodel --imdb VEDAI_1024_Train --cfg experiments/cfgs/vedai_1024.yml --solver models/vedai/VGG16_anchor_3_scale_conv_3_3/faster_rcnn_end2end/solver.prototxt --iters 60000

./tools/test_net.py --gpu 0 --def models/vedai/VGG16_anchor_3_scale_conv_3_3/faster_rcnn_end2end/test.prototxt --net output/vedai1024/vgg16_faster_rcnn_iter_60000.caffemodel --cfg experiments/cfgs/vedai_1024.yml --imdb VEDAI_1024_Test





