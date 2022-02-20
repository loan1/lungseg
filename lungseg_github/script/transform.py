import torchvision.transforms as transforms

image_t = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.5],  std = [0.5])    
])

mask_t = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])