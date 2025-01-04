import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import yaml
from pathlib import Path

class Text2ShapeModel(nn.Module):
    def __init__(self, config_path=None, device='cuda'):
        super().__init__()
        
        # Varsayılan konfigürasyon yolu
        if config_path is None:
            config_path = Path("models/text2shape/model_config.yaml")
            
        # Konfigürasyonu yükle
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.device = device
        
        # Text encoder (BERT)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config['model']['text_encoder'])
        self.text_encoder = AutoModel.from_pretrained(self.config['model']['text_encoder'])
        
        # Text projector
        self.text_projector = nn.Linear(
            self.text_encoder.config.hidden_size,
            self.config['architecture']['embedding_dim']
        )
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.config['architecture']['hidden_dim'],
            nhead=self.config['architecture']['num_heads'],
            dim_feedforward=self.config['architecture']['hidden_dim'] * 4,
            dropout=self.config['architecture']['dropout']
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.config['architecture']['num_layers']
        )
        
        # Point decoder
        self.num_points = self.config['generation']['num_points']
        self.grid_size = int((self.num_points ** 0.5))
        
        # Folding-based decoder
        self.folding1 = nn.Sequential(
            nn.Linear(self.config['architecture']['embedding_dim'] + 2, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 3)
        )
        
        self.folding2 = nn.Sequential(
            nn.Linear(self.config['architecture']['embedding_dim'] + 3, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 3)
        )
        
    def create_grid(self, batch_size):
        """2D grid oluştur"""
        x = torch.linspace(-1, 1, self.grid_size)
        y = torch.linspace(-1, 1, self.grid_size)
        grid_x, grid_y = torch.meshgrid(x, y)
        grid = torch.stack([grid_x, grid_y], dim=-1)
        grid = grid.reshape(-1, 2)
        grid = grid.unsqueeze(0).repeat(batch_size, 1, 1)
        return grid.to(self.device)
        
    def encode_text(self, text):
        """Metni kodla"""
        # Tokenize
        inputs = self.tokenizer(
            text,
            padding=self.config['text_processing']['padding'],
            truncation=self.config['text_processing']['truncation'],
            max_length=self.config['text_processing']['max_length'],
            return_tensors="pt"
        ).to(self.device)
        
        # Encode
        outputs = self.text_encoder(**inputs)
        text_features = outputs.last_hidden_state
        
        # Project
        text_features = self.text_projector(text_features)
        
        return text_features
        
    def forward(self, text):
        # Text encoding
        text_features = self.encode_text(text)
        
        # Transformer encoding
        transformer_out = self.transformer(text_features)
        
        # Point cloud generation
        batch_size = text_features.size(0)
        grid = self.create_grid(batch_size)
        
        # First folding
        features = transformer_out.mean(dim=1).unsqueeze(1).repeat(1, self.grid_size**2, 1)
        fold1_input = torch.cat([features, grid], dim=-1)
        fold1_out = self.folding1(fold1_input)
        
        # Second folding
        fold2_input = torch.cat([features, fold1_out], dim=-1)
        points = self.folding2(fold2_input)
        
        return points
        
    def save_model(self, path):
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.config
        }, path)
        
    def load_model(self, path):
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.config = checkpoint['config']
        self.eval() 