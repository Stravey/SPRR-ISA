import os
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision.transforms as transforms

# ===== 自定义 Dataset =====
class ImageFolderDataset(Dataset):
    def __init__(self, root_dir):
        """
        root_dir: 包含所有图片的文件夹路径
        """
        self.root_dir = root_dir
        self.image_files = [os.path.join(root_dir, f) 
                            for f in os.listdir(root_dir) 
                            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

        # 变换：Resize 到 256x256，并把像素缩放到 [0,1]
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()  # 会自动把像素缩放到 [0,1]
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        img = Image.open(img_path).convert("RGB")  # 转三通道
        img = self.transform(img)
        return img

# ===== 使用 DataLoader =====
if __name__ == "__main__":
    dataset = ImageFolderDataset("/home/data/guo/dataset/kaggledr/train/")
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=1)

    # 测试
    for batch in dataloader:
        print(batch.shape)  # [B, 3, 256, 256]
        break
