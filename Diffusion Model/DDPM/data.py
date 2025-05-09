from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler


class CelebADataset(Dataset):

    def __init__(self, root, img_shape=(64, 64)):
        super().__init__()
        self.root = root
        self.img_shape = img_shape
        self.filenames = sorted(os.listdir(root))

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, index: int):
        path = os.path.join(self.root, self.filenames[index])
        img = Image.open(path)
        pipeline = transforms.Compose([
            transforms.CenterCrop(168),
            transforms.Resize(self.img_shape),
            transforms.ToTensor()
        ])
        return pipeline(img)


CELEBA_HQ_DIR = '/kaggle/input/celebahq-resized-256x256/celeba_hq_256'
def get_dataloader(type, batch_size, img_shape=None, dist_train=False,
                   num_workers=4, use_lmdb=False, **kwargs):

    if img_shape is not None:
        kwargs['img_shape'] = img_shape
    dataset = CelebADataset(CELEBA_HQ_DIR, **kwargs)
    if dist_train:  # distributed training, always with data parallel
        sampler = DistributedSampler(dataset)
        dataloader = DataLoader(dataset, batch_size=batch_size,
                                sampler=sampler, num_workers=num_workers)
        return dataloader, sampler
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size,
                                shuffle=True, num_workers=num_workers)
        return dataloader
      
