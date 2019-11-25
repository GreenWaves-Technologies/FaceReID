import json
from PIL import Image

from torchreid.data_manager import build_transforms


def inference(model, image_path, args, use_gpu):
    model.eval()
    transforms = build_transforms(args.height, args.width, False, args.grayscale, args.no_normalize, args.quantization, args.bits)
    img = Image.open(image_path).convert('RGB')
    tensor = transforms(img)
    if use_gpu:
        tensor = tensor.cuda()
    embedding = model(tensor.unsqueeze_(0)).data.cpu().tolist()
    with open(image_path[:image_path.rfind('.')] + '.json', 'w') as f:
        json.dump(embedding, f)