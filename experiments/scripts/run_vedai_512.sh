/bash

LOG="../logs/vedai_512.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

cd ../../

./tools/train_net.py --gpu 0 --weights models/initialization/VGG_ILSVRC_16_layers.caffemodel --imdb VEDAI_512_Train --cfg experiments/cfgs/vedai_512.yml --solver models/vedai/VGG16_anchor_3_scale_conv_3_3/faster_rcnn_end2end/solver.prototxt --iters 60000

./tools/test_net.py --gpu 0 --def models/vedai/VGG16_anchor_3_scale_conv_3_3/faster_rcnn_end2end/test.prototxt --net output/vedai512/vgg16_faster_rcnn_iter_60000.caffemodel --cfg experiments/cfgs/vedai_512.yml --imdb VEDAI_512_Test





