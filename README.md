# Computer rendering project

## Command line to run 3D pose estimation

1. Run Detectron2
  ```bash
  cd inference
  python infer_video_d2.py --cfg COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml --output-dir output_video --image-ext mp4 ../input_video/
  ```

2. Create custom dataset
  ```bash
    cd ../data
    python prepare_data_2d_custom.py -i ../output_video/ -o myvideo
  ```

3. Run 3D pose estimation (export .npy)
   ```
   cd ../
   python run.py -d custom -k myvideo -arc 3,3,3,3,3 -c checkpoint --evaluate pretrained_h36m_detectron_coco.bin --render --viz-subject test.mp4 --viz-action custom --viz-camera 0 --viz-video input_video/test.mp4 --viz-output output.mp4 --viz-export test --viz-size 6
   ```
