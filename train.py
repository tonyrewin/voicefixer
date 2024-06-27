import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from voicefixer import VoiceFixer  # Импортируем VoiceFixer
from voicefixer.tools.wav import load_wave

# Define the dataset class
class VoiceDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.files = [f for f in os.listdir(data_dir) if f.endswith('.wav')]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.files[idx])
        audio, sr = load_wave(file_path)  # Получаем также частоту дискретизации
        return torch.tensor(audio).float(), sr

# Define the training module
class VoiceFixerTrainer(pl.LightningModule):
    def __init__(self, sample_rate=44100):
        super(VoiceFixerTrainer, self).__init__()
        self.voice_fixer = VoiceFixer(sample_rate=sample_rate)  # Инициализируем VoiceFixer
        self.sample_rate = sample_rate

    def forward(self, x):
        return self.voice_fixer.restore(x, self.sample_rate)  # Используем метод restore

    def training_step(self, batch, batch_idx):
        audio, sr = batch
        restored_audio = self(audio)
        loss = self.compute_loss(restored_audio, audio)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        audio, sr = batch
        restored_audio = self(audio)
        loss = self.compute_loss(restored_audio, audio)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.002)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
        return [optimizer], [scheduler]

    def compute_loss(self, output, target):
        return nn.MSELoss()(output, target)

# Set up data loaders
train_dataset = VoiceDataset(data_dir='data/train')
val_dataset = VoiceDataset(data_dir='data/valid')
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

# Initialize trainer
trainer = pl.Trainer(max_epochs=100, gpus=1)

# Initialize model
model = VoiceFixerTrainer()

# Train the model
trainer.fit(model, train_loader, val_loader)
