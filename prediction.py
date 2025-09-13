# prediction.py
from typing import Dict, List, Tuple, Optional
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import timm

# ===============================
# Config / Registry
# ===============================

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

MODEL_REGISTRY: Dict[str, Dict] = {
    "convnext_tiny.fb_in22k_ft_in1k": {
        "timm_name": "convnext_tiny.fb_in22k_ft_in1k",
        "img_size": 256,
    },
    "efficientnet_b0.ra_in1k": {
        "timm_name": "efficientnet_b0.ra_in1k",
        "img_size": 256,
    },
    "mobilenetv3_large_100.ra_in1k": {
        "timm_name": "mobilenetv3_large_100.ra_in1k",
        "img_size": 256,
    },
    "ecaresnet50d.miil_in1k": {
        "timm_name": "ecaresnet50d.miil_in1k",  # หรือ "ecaresnet50d"
        "img_size": 256,
    },
    "inception_v3.tf_in1k": {
        "timm_name": "inception_v3.tf_in1k",
        "img_size": 256,
    },
}

# ===============================
# Model build / load
# ===============================

def build_model(model_key: str, num_classes: int) -> torch.nn.Module:
    cfg = MODEL_REGISTRY[model_key]
    model = timm.create_model(cfg["timm_name"], num_classes=num_classes, pretrained=False)
    return model


def load_checkpoint_flex(model: torch.nn.Module, ckpt_path: str) -> None:
    """
    รองรับรูปแบบเช็คพอยต์ที่พบบ่อย:
    - torch.save(model)                             → obj เป็น nn.Module
    - torch.save(model.state_dict())               → obj เป็น state_dict
    - dict ที่มี key 'state_dict' หรือ 'model'     → เก็บ state_dict หรือโมเดลไว้ข้างใน
    """
    # PyTorch 2.6 เปลี่ยน default ของ weights_only → ต้องกำหนดเป็น False เพื่อรองรับไฟล์เก่า
    try:
        obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        obj = torch.load(ckpt_path, map_location="cpu")

    state_dict = None
    if isinstance(obj, torch.nn.Module):
        state_dict = obj.state_dict()
    elif isinstance(obj, dict):
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            state_dict = obj["state_dict"]
        elif "model" in obj:
            if isinstance(obj["model"], dict):
                state_dict = obj["model"]
            elif isinstance(obj["model"], torch.nn.Module):
                state_dict = obj["model"].state_dict()
        if state_dict is None:
            # เดาว่าเป็น state_dict ตรง ๆ
            state_dict = obj
    else:
        # ฟอร์แมตไม่รู้จัก
        raise ValueError(f"Unsupported checkpoint format at: {ckpt_path}")

    # ล้าง prefix ที่มาจาก DataParallel หรือ wrapper อื่น ๆ
    cleaned = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[len("module."):]
        if k.startswith("model."):
            k = k[len("model."):]
        cleaned[k] = v

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing or unexpected:
        print("[checkpoint] missing keys:", missing)
        print("[checkpoint] unexpected keys:", unexpected)

    model.eval()


# ===============================
# Transform / Inference
# ===============================

def make_transform(img_size: int):
    # ใช้ torchvision FAPI แบบไลท์เวท
    from torchvision.transforms.functional import resize, to_tensor, normalize
    def _tfm(pil_img: Image.Image) -> torch.Tensor:
        img = pil_img.convert("RGB")
        # รีไซซ์เป็นสี่เหลี่ยมจัตุรัส (ถ้าชุดเทรนใช้สี่เหลี่ยมอยู่แล้วจะตรงกันพอดี)
        img = resize(img, [img_size, img_size], antialias=True)
        x = to_tensor(img)
        x = normalize(x, mean=IMAGENET_MEAN, std=IMAGENET_STD)
        return x
    return _tfm


def predict_probs(
    model: torch.nn.Module,
    image: Image.Image,
    img_size: int,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """คืนค่า probs รูปทรง (1, C)"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)

    x = make_transform(img_size)(image).unsqueeze(0).float().to(device)
    with torch.inference_mode():
        logits = model(x)
        probs = F.softmax(logits, dim=1).cpu().numpy()
    return probs


def topk_from_probs(probs: np.ndarray, labels: List[str], k: int = 5):
    """คืนลิสต์ [(label, prob)] ของ top-k จากมากไปน้อย"""
    p = probs[0]
    k = min(k, len(p))
    idx = np.argsort(-p)[:k]
    return [(labels[i], float(p[i])) for i in idx]
