"""
An example that uses TensorRT's Python api to make inferences.
"""
import ctypes
import os
import random
import sys
import threading
import time

import cv2
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt
import torch
import torchvision

import pdb

INPUT_W = 416
INPUT_H = 416 
CONF_THRESH = 0
IOU_THRESHOLD = 0.4


def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    """
    description: Plots one bounding box on image img,
                 this function comes from YoLov5 project.
    param: 
        x:      a box likes [x1,y1,x2,y2]
        img:    a opencv image object
        color:  color to draw rectangle, such as (0,255,0)
        label:  str
-------------------------------------------------------------------
PyCUDA ERROR: The context stack was not empty upon module cleanup.
-------------------------------------------------------------------
A context was still active when the context stack was being
cleaned up. At this point in our execution, CUDA may already
have been deinitialized, so there is no way we can finish
cleanly. The program will be aborted now.
Use Context.pop() to avoid this problem.
-------------------------------------------------------------------
Aborted (core dumped)        line_thickness: int
    return:
        no return

    """
    tl = (
        line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    )  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(
            img,
            label,
            (c1[0], c1[1] - 2),
            0,
            tl / 3,
            [225, 255, 255],
            thickness=tf,
            lineType=cv2.LINE_AA,
        )


class YoLov5TRT(object):
    """
    description: A YOLOv5 class that warps TensorRT ops, preprocess and postprocess ops.
    """

    def __init__(self,PLUGIN_LIBRARY,  engine_file_path, categories):
        self.IOUThr = 1
        self.Thr = 0
        self.categories = categories
        # Create a Context on this device,
        #PLUGIN_LIBRARY = "build/libmyplugins.so"
        ctypes.CDLL(PLUGIN_LIBRARY)
        #engine_file_path = "build/yolov5s.engine"
        self.cfx = cuda.Device(0).make_context()
        stream = cuda.Stream()
        TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        runtime = trt.Runtime(TRT_LOGGER)

        # Deserialize the engine from file
        with open(engine_file_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        context = engine.create_execution_context()

        host_inputs = []
        cuda_inputs = []
        host_outputs = []
        cuda_outputs = []
        bindings = []

        for binding in engine:
            size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings.
            bindings.append(int(cuda_mem))
            # Append to the appropriate list.
            if engine.binding_is_input(binding):
                host_inputs.append(host_mem)
                cuda_inputs.append(cuda_mem)
            else:
                host_outputs.append(host_mem)
                cuda_outputs.append(cuda_mem)

        # Store
        self.stream = stream
        self.context = context
        self.engine = engine
        self.host_inputs = host_inputs
        self.cuda_inputs = cuda_inputs
        self.host_outputs = host_outputs
        self.cuda_outputs = cuda_outputs
        self.bindings = bindings

    def infer(self, RGB, show_image = 0):
        overall_tic = time.time()
        #print("Start _infer")
        threading.Thread.__init__(self)
        # Make self the active context, pushing it on top of the context stack.
        self.cfx.push()
        # Restore
        stream = self.stream
        context = self.context
        engine = self.engine
        host_inputs = self.host_inputs
        cuda_inputs = self.cuda_inputs
        host_outputs = self.host_outputs
        cuda_outputs = self.cuda_outputs
        bindings = self.bindings
        # Do image preprocess
        pre_tic = time.time()
        input_image, image_raw, origin_h, origin_w = self.preprocess_image(
           RGB 
        )
        pre_toc = time.time() - pre_tic

        infe_tic = time.time()
        # Copy input image to host buffer
        np.copyto(host_inputs[0], input_image.ravel())
        # Transfer input data  to the GPU.
        cuda.memcpy_htod_async(cuda_inputs[0], host_inputs[0], stream)
        #print("Trans DOne")
        # Run inference.
        context.execute_async(bindings=bindings, stream_handle=stream.handle)
        # Transfer predictions back from the GPU.
        cuda.memcpy_dtoh_async(host_outputs[0], cuda_outputs[0], stream)
        #print("Cuda Done")
        # Synchronize the stream
        stream.synchronize()
        # Remove any context from the top of the context stack, deactivating it.
        self.cfx.pop()
        infe_toc = time.time() - infe_tic

        post_tic = time.time()
        #print("Start1 DOne")
        # Here we use the first row of output in that batch_size = 1
        output = host_outputs[0]
        #print(output)
        #print(type(output))
        #print(output.shape)
        # Do postprocess
        #print("Start Inference")
        num_box = output[0]
        output = output[1:]
        output = output.reshape((-1,6))
        #print(output)
        result_boxes, result_scores, result_classid = self.post_process(
            num_box, output, origin_h, origin_w
        )
        KeepList = self.Filter(result_boxes,result_scores)
        print(KeepList)
        result_boxes, result_scores, result_classid = result_boxes[KeepList], result_scores[KeepList], result_classid[KeepList]
        #pdb.set_trace()


        post_toc = time.time() - post_tic
        #print("Inference Done")
        draw_tic = time.time()
        # Draw rectangles and labels on the original image

        for i in range(len(result_boxes)):
            box = result_boxes[i]
            plot_one_box(
                box,
                image_raw,
                label="{}:{:.2f}".format(
                    self.categories[int(result_classid[i])], result_scores[i]
                ),
            )

        draw_toc = time.time() - draw_tic
        overall_toc = time.time() - overall_tic
        print("FPS")
        print(1 / overall_toc)
        with open("Target.txt", 'a') as Tar:
            Tar.write(str(pre_toc)+'\n'
                    +str(infe_toc)+'\n'
                    +str(post_toc)+'\n'
                    +str(draw_toc)+'\n'
                    +str(overall_toc)+'\n')
            
        if(show_image == 1):
            cv2.imshow("Frame", image_raw)
            cv2.waitKey(1)
        """
        parent, filename = os.path.split(input_image_path)
        save_name = os.path.join(parent, "output_" + filename)
        # 　Save image
        cv2.imwrite(save_name, image_raw)
        """
    def destroy(self):
        # Remove any context from the top of the context stack, deactivating it.
        self.cfx.pop()

    def preprocess_image(self, RGB):
        """
        description: Read an image from image path, convert it to RGB,
                     resize and pad it to target size, normalize to [0,1],
                     transform to NCHW format.
        param:
            input_image_path: str, image path
        return:
            image:  the processed image
            image_raw: the original image
            h: original height
            w: original width
        """
        image_raw = RGB 
        h, w, c = image_raw.shape
        image = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)
        # Calculate widht and height and paddings
        r_w = INPUT_W / w
        r_h = INPUT_H / h
        if r_h > r_w:
            tw = INPUT_W
            th = int(r_w * h)
            tx1 = tx2 = 0
            ty1 = int((INPUT_H - th) / 2)
            ty2 = INPUT_H - th - ty1
        else:
            tw = int(r_h * w)
            th = INPUT_H
            tx1 = int((INPUT_W - tw) / 2)
            tx2 = INPUT_W - tw - tx1
            ty1 = ty2 = 0
        # Resize the image with long side while maintaining ratio
        image = cv2.resize(image, (tw, th))
        # Pad the short side with (128,128,128)
        image = cv2.copyMakeBorder(
            image, ty1, ty2, tx1, tx2, cv2.BORDER_CONSTANT, (128, 128, 128)
        )
        image = image.astype(np.float32)
        # Normalize to [0,1]
        image /= 255.0
        # HWC to CHW format:
        image = np.transpose(image, [2, 0, 1])
        # CHW to NCHW format
        image = np.expand_dims(image, axis=0)
        # Convert the image to row-major order, also known as "C order":
        image = np.ascontiguousarray(image)
        return image, image_raw, h, w

    def xywh2xyxy(self, origin_h, origin_w, x):
        """
        description:    Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        param:
            origin_h:   height of original image
            origin_w:   width of original image
            x:          A boxes tensor, each row is a box [center_x, center_y, w, h]
        return:
            y:          A boxes tensor, each row is a box [x1, y1, x2, y2]
        """
        y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
        r_w = INPUT_W / origin_w
        r_h = INPUT_H / origin_h
        if r_h > r_w:
            y[:, 0] = x[:, 0] - x[:, 2] / 2
            y[:, 2] = x[:, 0] + x[:, 2] / 2
            y[:, 1] = x[:, 1] - x[:, 3] / 2 - (INPUT_H - r_w * origin_h) / 2
            y[:, 3] = x[:, 1] + x[:, 3] / 2 - (INPUT_H - r_w * origin_h) / 2
            y /= r_w
        else:
            y[:, 0] = x[:, 0] - x[:, 2] / 2 - (INPUT_W - r_h * origin_w) / 2
            y[:, 2] = x[:, 0] + x[:, 2] / 2 - (INPUT_W - r_h * origin_w) / 2
            y[:, 1] = x[:, 1] - x[:, 3] / 2
            y[:, 3] = x[:, 1] + x[:, 3] / 2
            y /= r_h

        return y


    def post_process(self, num_box, output, origin_h, origin_w):
        num_box = int(num_box)
        """
        description: postprocess the prediction
        param:
            output:     A tensor likes [num_boxes,cx,cy,w,h,conf,cls_id, cx,cy,w,h,conf,cls_id, ...] 
            origin_h:   height of original image
            origin_w:   width of original image
        return:
            result_boxes: finally boxes, a boxes tensor, each row is a box [x1, y1, x2, y2]
            result_scores: finally scores, a tensor, each element is the score correspoing to box
            result_classid: finally classid, a tensor, each element is the classid correspoing to box
        """
        #print("Get in post process")
        # Get the num of boxes detected
        """
        # Reshape to a two dimentional ndarray
        pred = np.reshape(output[1:], (-1, 6))[:num, :]
        # to a torch Tensor
        pred = torch.Tensor(pred).cuda()
        """
        # Get the boxes
        boxes = output[0:num_box, :4]
        # Get the scores
        scores = output[0:num_box, 4]
        # Get the classid
        classid = output[0:num_box, 5]
        #print("Value Given done")
        #print("Given DOne")
        # Choose those boxes that score > CONF_THRESH
        si = scores > CONF_THRESH
        boxes = boxes[si, :]
        scores = scores[si]
        classid = classid[si]
        tic = time.time()
        boxes = torch.Tensor(boxes)
        scores = torch.Tensor(scores)
        classid = torch.Tensor(classid)
        """
        print("Start CUDA")
        boxes = torch.Tensor(boxes).cuda()
        print("CUDA1 Done")
        scores = torch.Tensor(scores).cuda()
        classid = torch.Tensor(classid).cuda()
        print("CUDA Done")
        """
        # Trandform bbox from [center_x, center_y, w, h] to [x1, y1, x2, y2]
        boxes = self.xywh2xyxy(origin_h, origin_w, boxes)
        #print("Resign the Value Done")
        # Do nms
        #print("Start NMS")

        indices = torchvision.ops.nms(boxes, scores, iou_threshold=IOU_THRESHOLD).cpu()
        #print("NMS Done")
        result_boxes = boxes[indices, :]
        result_scores = scores[indices]
        result_classid = classid[indices]
        #print("NMS Total used in", time.time() - tic)
        return result_boxes, result_scores, result_classid

    def Filter(self, pos, pen):
        NUM = len(pos)
        print(NUM)
        #pdb.set_trace()
        Keep_List = []

        for i in range(NUM):
            if(pen[i] < self.Thr):
                continue
            for j in range(NUM):
                if(i == j):
                    continue
                IOU = self.IOU(pos[i],pos[j])
                if(IOU >  self.IOUThr):
                    if(pen[i] > pen[j]):
                        Keep_List.append(i)
                else:
                    Keep_List.append(i)
        Keep_List = list(set(Keep_List))
        return Keep_List

    def IOU(self,pos1,pos2):
        [x1,y1,x2,y2] = pos1
        [x3,y3,x4,y4] = pos2
        w1 = x2 - x1
        w2 = x4 - x3
        h1 = y2 - y1
        h2 = y4 - y3
        IOU_W = min(x1,x2,x3,x4) + w1 + w2 - max(x1,x2,x3,x4)
        IOU_H = min(y1,y2,y3,y4) + h1 + h2 - max(y1,y2,y3,y4)
        IOU_S = IOU_W * IOU_H
        P = 1.0 * IOU_S / (w1 * h1 + w2 * h2 - IOU_S) + 0.000001
        return P



if __name__ == "__main__":
    # load custom plugins
    # load coco labels

    categories = ["CA001", "CA002", "CA003", "CA004", "CB001", "CB002", "CB003", "CB004", "CC001", "CC002",
            "CC003", "CC004", "CD001", "CD002", "CD003", "CD004", "dog", "horse", "sheep", "cow",
            "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
            "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
            "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
            "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
            "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
            "hair drier", "toothbrush"]

    PLUGIN_LIBRARY = "build/libmyplugins.so"
    engine_file_path = "build/yolov5s.engine"

    # a  YoLov5TRT instance
    tic_init = time.time()
    cap = cv2.VideoCapture(0)
    yolov5_wrapper = YoLov5TRT(PLUGIN_LIBRARY, engine_file_path,categories)
    with open("init.txt", 'w') as init_txt:
        init_txt.write(str(time.time() - tic_init))
    #print("init Done")
    
    # from https://github.com/ultralytics/yolov5/tree/master/inference/images
    vedio = cv2.VideoCapture("CER.mp4")
    ret = 1
    RGB_Org = cv2.imread('img_160.png')
    while(ret):
        # creddate a new thread to do inferenceo
        #ret, RGB = vedio.read()
        RGB_Org = cv2.imread('img_160.png')

        RGB = RGB_Org
        yolov5_wrapper.infer(RGB,1)

    # destroy the instance
    yolov5_wrapper.destroy()

