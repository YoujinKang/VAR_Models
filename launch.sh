# docker run --gpus all -v /hdd/user4/data/vg_raw_images:/workspace/images:ro -v /hdd/user4/workspace/ExtractFeatures:/workspace/features --rm -it airsplay/bottom-up-attention bash
docker run --gpus all -v /hdd/user16/HT/data/vcr/vcr1images/vcr1images:/workspace/images:ro -v /hdd/user4/workspace/ExtractFeatures:/workspace/features --rm -it airsplay/bottom-up-attention bash