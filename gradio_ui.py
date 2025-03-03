import torch
from torch.nn import functional as F
import numpy as np
import gradio as gr
import os
import shutil
import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

def load_checkpoint(checkpoint):
    shutil.rmtree("./train_log", ignore_errors=True)
    shutil.copytree(checkpoint, "./train_log")
    checkpoint = "train_log"
    try:
        del Model
    except:
        pass
    global model
    try:
        try:
            try:
                from model.RIFE_HDv2 import Model
                model = Model()
                model.load_model(checkpoint, -1)
                gr.Info("Loaded v2.x HD model.")
            except:
                from train_log.RIFE_HDv3 import Model
                model = Model()
                model.load_model(checkpoint, -1)
                gr.Info("Loaded v3.x HD model.")
        except:
            from model.RIFE_HD import Model
            model = Model()
            model.load_model(checkpoint, -1)
            gr.Info("Loaded v1.x HD model")
    except:
        from model.RIFE import Model
        model = Model()
        model.load_model(checkpoint, -1)
        gr.Info("Loaded ArXiv-RIFE model")
    model.eval()
    model.device()
    return

def infer(frame1, frame2, exp, ratio, rthreshold, rmaxcycles):
    img0 = np.array(frame1)
    img1 = np.array(frame2)
    img0 = (torch.tensor(img0.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)
    img1 = (torch.tensor(img1.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)
    n, c, h, w = img0.shape
    ph = ((h - 1) // 32 + 1) * 32
    pw = ((w - 1) // 32 + 1) * 32
    padding = (0, pw - w, 0, ph - h)
    img0 = F.pad(img0, padding)
    img1 = F.pad(img1, padding)

    if ratio:
        img_list = [img0]
        img0_ratio = 0.0
        img1_ratio = 1.0
        if ratio <= img0_ratio + rthreshold / 2:
            middle = img0
        elif ratio >= img1_ratio - rthreshold / 2:
            middle = img1
        else:
            tmp_img0 = img0
            tmp_img1 = img1
            for inference_cycle in range(rmaxcycles):
                middle = model.inference(tmp_img0, tmp_img1)
                middle_ratio = ( img0_ratio + img1_ratio ) / 2
                if ratio - (rthreshold / 2) <= middle_ratio <= ratio + (rthreshold / 2):
                    break
                if ratio > middle_ratio:
                    tmp_img0 = middle
                    img0_ratio = middle_ratio
                else:
                    tmp_img1 = middle
                    img1_ratio = middle_ratio
        img_list.append(middle)
        img_list.append(img1)
    else:
        img_list = [img0, img1]
        for i in range(exp):
            tmp = []
            for j in range(len(img_list) - 1):
                mid = model.inference(img_list[j], img_list[j + 1])
                tmp.append(img_list[j])
                tmp.append(mid)
            tmp.append(img1)
            img_list = tmp
    img_list = [(img[0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w] for img in img_list]
    return img_list

weights = os.listdir("weights")
weights = [os.path.join("weights", w) for w in weights]
model = None

with gr.Blocks() as demo:
    gr.Markdown(
    '''
    # RIFE Inference
    ''')
    with gr.Column():
        checkpoint = gr.Dropdown(choices=weights, label="Checkpoint")
        checkpoint_btn = gr.Button("Load")
        exp = gr.Number(minimum=0, value=1, label="Exp")
        ratio = gr.Slider(minimum=0.0, maximum=1.0, value=0.0, label="Inference Ratio", info="Inference ration between two images with 0-1 range")
        rthreshold = gr.Number(minimum=0.0, value=0.02, label="rthreshold", info="Returns image when actual ratio falls in given range threshold")
        rmaxcycles = gr.Number(minimum=0, value=8, label="rmaxcycles", info="Limit max number of bisectional cycles")
        with gr.Row():
            frame1 = gr.Image(type="pil", label="Frame 1", sources=['upload'])
            frame2 = gr.Image(type="pil", label="Frame 2", sources=['upload'])
    with gr.Column():
        frameI = gr.Gallery(type="pil", label="Synthesized Result", file_types=["images"])
        run_btn = gr.Button(value="Run")

    checkpoint_btn.click(fn=load_checkpoint, inputs=[checkpoint], outputs=None)
    run_btn.click(fn=infer, inputs=[frame1, frame2, exp, ratio, rthreshold, rmaxcycles], outputs=[frameI])

demo.launch(server_name="0.0.0.0", server_port=7068)
