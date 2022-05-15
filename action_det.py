import cv2
import math
import time
import random
import warnings
import argparse

import torch
import numpy as np
import pytorchvideo

import mmcv

from mmdet.apis import inference_detector, init_detector


from pytorchvideo.transforms.functional import (
    uniform_temporal_subsample,
    short_side_scale_with_boxes,
    clip_boxes_to_image,)
from torchvision.transforms._functional_video import normalize
from pytorchvideo.data.ava import AvaLabeledVideoFramePaths
from pytorchvideo.data.utils import thwc_to_cthw
from pytorchvideo.models.hub import slowfast_r50_detection

from deep_sort.deep_sort import DeepSort

from selfutils.encode_videos import EncodedVideo

import ipdb

warnings.filterwarnings("ignore",category=UserWarning)
    

def parse_args():
    parser = argparse.ArgumentParser(description='MMDetection video demo')

    parser.add_argument('--video', help='Video file')
    parser.add_argument('--config', help='Config file')
    parser.add_argument('--checkpoint', help='Checkpoint file')
    parser.add_argument('--imsize', type=int, default=224, help='inference size (pixels)')
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    parser.add_argument('--score-thr', type=float, default=0.3, help='Bbox score threshold')
    parser.add_argument('--out', default='output.mp4', type=str, help='Output video file')
    parser.add_argument('--wait-time', type=float, default=1, help='The interval of show (s), 0 is block')
    args = parser.parse_args()

    return args


def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y

def ava_inference_transform(clip, boxes,
    num_frames = 32, #if using slowfast_r50_detection, change this to 32, 4 for slow 
    crop_size = 640, 
    data_mean = [0.45, 0.45, 0.45], 
    data_std = [0.225, 0.225, 0.225],
    slow_fast_alpha = 4, #if using slowfast_r50_detection, change this to 4, None for slow
):
    boxes = np.array(boxes)
    roi_boxes = boxes.copy()
    clip = uniform_temporal_subsample(clip, num_frames)
    clip = clip.float()
    clip = clip / 255.0
    height, width = clip.shape[2], clip.shape[3]
    boxes = clip_boxes_to_image(boxes, height, width)
    clip, boxes = short_side_scale_with_boxes(clip,size=crop_size,boxes=boxes,)
    clip = normalize(clip,
        np.array(data_mean, dtype=np.float32),
        np.array(data_std, dtype=np.float32),) 
    boxes = clip_boxes_to_image(boxes, clip.shape[2],  clip.shape[3])
    if slow_fast_alpha is not None:
        fast_pathway = clip
        slow_pathway = torch.index_select(clip,1,
            torch.linspace(0, clip.shape[1] - 1, clip.shape[1] // slow_fast_alpha).long())
        clip = [slow_pathway, fast_pathway]
    
    return clip, torch.from_numpy(boxes), roi_boxes

def plot_one_box(x, img, color=[100,100,100], text_info="None",
                 velocity=None,thickness=1,fontsize=0.5,fontthickness=1):
    # Plots one bounding box on image img
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness, lineType=cv2.LINE_AA)
    t_size = cv2.getTextSize(text_info, cv2.FONT_HERSHEY_TRIPLEX, fontsize , fontthickness+2)[0]
    cv2.rectangle(img, c1, (c1[0] + int(t_size[0]), c1[1] + int(t_size[1]*1.45)), color, -1)
    cv2.putText(img, text_info, (c1[0], c1[1]+t_size[1]+2), 
                cv2.FONT_HERSHEY_TRIPLEX, fontsize, [255,255,255], fontthickness)
    return img

def save_yolopreds_tovideo(imgs, preds,id_to_ava_labels,color_map,output_video):
    for i, (im, pred) in enumerate(zip(imgs, preds)):

        if pred.shape[0]:
            for j, (*box, cls, trackid, vx, vy) in enumerate(pred):
                if int(cls) != 0:
                    ava_label = ''
                    continue   # only plot person

                elif trackid in id_to_ava_labels.keys():
                    ava_label = id_to_ava_labels[trackid].split(' ')[0]
                else:
                    ava_label = 'Unknow'

                text = '{} {} {}'.format(int(trackid),'person',ava_label)

                color = color_map[int(cls)]

                im = plot_one_box(box,im,color,text)

        output_video.write(im.astype(np.uint8))

def main():
    args = parse_args()

    model = init_detector(args.config, args.checkpoint, device=args.device)

    video = EncodedVideo(args.video) # decode video

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(
        args.out, fourcc, video.fps,
        (video.width, video.height))

    video_model = slowfast_r50_detection(True).eval().to(args.device)
    deepsort_tracker = DeepSort("deep_sort/deep_sort/deep/checkpoint/ckpt.t7")
    ava_labelnames,_ = AvaLabeledVideoFramePaths.read_label_map("selfutils/temp.pbtxt")
    coco_color_map = [[random.randint(0, 255) for _ in range(3)] for _ in range(80)]

    print("Video lenght: {} s, processing...".format(math.ceil(video.duration)))
    #roi_size = [0, int(0.3*width),int(0.6*height),int(0.6*width)]
    
    a=time.time()

    for ti in range(0, math.ceil(video.duration), 1):
        video_clips = video.get_clip(ti, ti+1-0.04)  # thwc
        if video_clips is None:
            continue
        
        #ipdb.set_trace()
        img_num= video_clips.shape[0]
        imgs=[]
        det_preds = []

        for j in range(img_num):
            # crop
            img = video_clips[j,:,:,:]
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            det = inference_detector(model, img)  # (tl_x, tl_y, br_x, br_y, score)
            det_preds.append(det)
            imgs.append(img)

        # numpy(t,h,w,c) -> Tensor(c,t,h,w)
        video_clips = torch.from_numpy(video_clips)
        video_clips = video_clips.to(torch.float32)
        video_clips = thwc_to_cthw(video_clips)

        print(ti,video_clips.shape, img_num)

        deepsort_outputs=[]
        for j in range(len(det_preds)):
            pred = det_preds[j][0]
            xywh = xyxy2xywh(pred[:,0:4])
            temp = deepsort_tracker.update(xywh, pred[:,4:5],[0]*len(pred),imgs[j])

            if len(temp)==0:
                temp=np.ones((0,8))

            deepsort_outputs.append(temp.astype(np.float32))

        id_to_ava_labels={}

        if deepsort_outputs[img_num//2].shape[0]:
            inputs,inp_boxes,_=ava_inference_transform(video_clips,deepsort_outputs[img_num//2][:,0:4],crop_size=args.imsize)
            
            inp_boxes = torch.cat([torch.zeros(inp_boxes.shape[0],1), inp_boxes], dim=1)

            if isinstance(inputs, list):
                inputs = [inp.unsqueeze(0).to(args.device) for inp in inputs]
            else:
                inputs = inputs.unsqueeze(0).to(args.device)

            with torch.no_grad():
                slowfaster_preds = video_model(inputs, inp_boxes.to(args.device))
                slowfaster_preds = slowfaster_preds.cpu()
            for tid,avalabel in zip(deepsort_outputs[img_num//2][:,5].tolist(),np.argmax(slowfaster_preds,axis=1).tolist()):
                id_to_ava_labels[tid]=ava_labelnames[avalabel+1]

        save_yolopreds_tovideo(imgs, deepsort_outputs, id_to_ava_labels,coco_color_map,video_writer)


    print("total cost: {:.3f}s, video clips length: {}s".format(time.time()-a,video.duration))
    
    video_writer.release()
    print('saved video to:', args.out)


if __name__ == '__main__':
    main()

# Todo: roi : https://www.cxymm.net/article/qq_36584673/116265950
# Todo: decord gpu : https://github.com/dmlc/decord