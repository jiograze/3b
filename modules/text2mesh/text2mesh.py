import torch
from typing import Dict, Optional
import openai
import numpy as np
import traceback

class Text2Mesh:
    def __init__(self, api_key: str):
        """
        Text2Mesh sınıfı
        
        Args:
            api_key (str): OpenAI API anahtarı
        """
        self.api_key = api_key
        openai.api_key = self.api_key
    
    def edit_geometry(self, prompt: str, points: torch.Tensor) -> torch.Tensor:
        """
        Metin promptu ile geometri düzenleme
        
        Args:
            prompt (str): Metin promptu
            points (torch.Tensor): Nokta bulutu (N, 3)
            
        Returns:
            torch.Tensor: Düzenlenmiş nokta bulutu
        """
        try:
            response = openai.Completion.create(
                engine="davinci-codex",
                prompt=f"Edit the geometry of the following 3D points based on the prompt: {prompt}\nPoints: {points.tolist()}",
                max_tokens=1000
            )
            edited_points = np.array(eval(response.choices[0].text.strip()))
            return torch.tensor(edited_points, dtype=points.dtype)
        except Exception as e:
            print(f"\nGeometri düzenleme hatası: {str(e)}")
            traceback.print_exc()
            raise
    
    def semantic_surface_editing(self, prompt: str, points: torch.Tensor) -> torch.Tensor:
        """
        Semantik yüzey düzenleme
        
        Args:
            prompt (str): Metin promptu
            points (torch.Tensor): Nokta bulutu (N, 3)
            
        Returns:
            torch.Tensor: Düzenlenmiş nokta bulutu
        """
        try:
            response = openai.Completion.create(
                engine="davinci-codex",
                prompt=f"Edit the surface of the following 3D points based on the prompt: {prompt}\nPoints: {points.tolist()}",
                max_tokens=1000
            )
            edited_points = np.array(eval(response.choices[0].text.strip()))
            return torch.tensor(edited_points, dtype=points.dtype)
        except Exception as e:
            print(f"\nSemantik yüzey düzenleme hatası: {str(e)}")
            traceback.print_exc()
            raise
    
    def style_transfer(self, prompt: str, points: torch.Tensor) -> torch.Tensor:
        """
        Stil transferi
        
        Args:
            prompt (str): Metin promptu
            points (torch.Tensor): Nokta bulutu (N, 3)
            
        Returns:
            torch.Tensor: Stil transferi uygulanmış nokta bulutu
        """
        try:
            response = openai.Completion.create(
                engine="davinci-codex",
                prompt=f"Apply style transfer to the following 3D points based on the prompt: {prompt}\nPoints: {points.tolist()}",
                max_tokens=1000
            )
            styled_points = np.array(eval(response.choices[0].text.strip()))
            return torch.tensor(styled_points, dtype=points.dtype)
        except Exception as e:
            print(f"\nStil transferi hatası: {str(e)}")
            traceback.print_exc()
            raise
    
    def add_details(self, prompt: str, points: torch.Tensor) -> torch.Tensor:
        """
        Detay ekleme
        
        Args:
            prompt (str): Metin promptu
            points (torch.Tensor): Nokta bulutu (N, 3)
            
        Returns:
            torch.Tensor: Detay eklenmiş nokta bulutu
        """
        try:
            response = openai.Completion.create(
                engine="davinci-codex",
                prompt=f"Add details to the following 3D points based on the prompt: {prompt}\nPoints: {points.tolist()}",
                max_tokens=1000
            )
            detailed_points = np.array(eval(response.choices[0].text.strip()))
            return torch.tensor(detailed_points, dtype=points.dtype)
        except Exception as e:
            print(f"\nDetay ekleme hatası: {str(e)}")
            traceback.print_exc()
            raise
    
    def remove_details(self, prompt: str, points: torch.Tensor) -> torch.Tensor:
        """
        Detay çıkarma
        
        Args:
            prompt (str): Metin promptu
            points (torch.Tensor): Nokta bulutu (N, 3)
            
        Returns:
            torch.Tensor: Detay çıkarılmış nokta bulutu
        """
        try:
            response = openai.Completion.create(
                engine="davinci-codex",
                prompt=f"Remove details from the following 3D points based on the prompt: {prompt}\nPoints: {points.tolist()}",
                max_tokens=1000
            )
            simplified_points = np.array(eval(response.choices[0].text.strip()))
            return torch.tensor(simplified_points, dtype=points.dtype)
        except Exception as e:
            print(f"\nDetay çıkarma hatası: {str(e)}")
            traceback.print_exc()
            raise
