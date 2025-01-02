"""Main UI module for Ötüken3D."""

import streamlit as st
import torch
from typing import Dict, Any
from pathlib import Path
import tempfile
import os
import time

from ..core.logger import setup_logger
from ..model3d_integration.text_to_shape import TextToShape
from ..model3d_integration.image_to_shape import ImageToShape

logger = setup_logger(__name__)

def setup_sidebar():
    """Setup sidebar configuration."""
    st.sidebar.title("⚙️ Ayarlar")
    
    # Model seçimi
    model_type = st.sidebar.selectbox(
        "Model Tipi",
        ["Metin-to-3D", "Görsel-to-3D"]
    )
    
    # GPU/CPU seçimi
    device_type = st.sidebar.radio(
        "İşlem Birimi",
        ["GPU", "CPU"],
        disabled=not torch.cuda.is_available()
    )
    
    # Gelişmiş ayarlar
    with st.sidebar.expander("Gelişmiş Ayarlar"):
        voxel_resolution = st.slider(
            "Voksel Çözünürlüğü",
            min_value=32,
            max_value=256,
            value=128,
            step=32
        )
        
        mesh_simplification = st.slider(
            "Mesh Sadeleştirme",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1
        )
        
        use_diffusion = st.checkbox(
            "Diffusion İyileştirmesi",
            value=False
        )
    
    return {
        "model_type": model_type,
        "device": torch.device("cuda" if device_type == "GPU" and torch.cuda.is_available() else "cpu"),
        "voxel_resolution": voxel_resolution,
        "mesh_simplification": mesh_simplification,
        "use_diffusion": use_diffusion
    }

def setup_text_to_3d_ui(config: Dict[str, Any], model: TextToShape):
    """Setup text-to-3D UI."""
    st.title("🎨 Metin-to-3D Dönüşüm")
    
    # Metin girişi
    text_input = st.text_area(
        "3D Model Açıklaması",
        placeholder="Örnek: Geleneksel Türk motifli bir vazo",
        help="Oluşturmak istediğiniz 3D modelin detaylı açıklamasını girin"
    )
    
    # Stil seçenekleri
    style_options = st.multiselect(
        "Stil Seçenekleri",
        ["Geleneksel", "Modern", "Minimalist", "Detaylı", "Sade"],
        default=["Geleneksel"]
    )
    
    # Oluştur butonu
    if st.button("3D Model Oluştur", disabled=not text_input):
        with st.spinner("3D model oluşturuluyor..."):
            try:
                # Stil bilgisini metne ekle
                full_text = f"{text_input} (Stil: {', '.join(style_options)})"
                
                # Model çıktısını al
                mesh = model.predict(full_text)
                
                # Geçici dosya oluştur
                with tempfile.NamedTemporaryFile(suffix=".obj", delete=False) as tmp:
                    mesh.export(tmp.name)
                    
                    # 3D viewer ile göster
                    st.success("3D model başarıyla oluşturuldu!")
                    st.write("Oluşturulan model:")
                    st.components.v1.iframe(
                        f"https://3dviewer.net/embed.html?model={tmp.name}",
                        height=400
                    )
                    
                    # İndirme butonu
                    with open(tmp.name, "rb") as f:
                        st.download_button(
                            "3D Modeli İndir (OBJ)",
                            f,
                            file_name="model.obj",
                            mime="model/obj"
                        )
                    
                # Geçici dosyayı temizle
                os.unlink(tmp.name)
                
            except Exception as e:
                st.error(f"Model oluşturma hatası: {str(e)}")
                logger.error(f"Model generation error: {str(e)}")

def setup_image_to_3d_ui(config: Dict[str, Any], model: ImageToShape):
    """Setup image-to-3D UI."""
    st.title("📸 Görsel-to-3D Dönüşüm")
    
    # Görsel yükleme
    uploaded_files = st.file_uploader(
        "Referans Görseller",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
        help="Bir veya daha fazla referans görsel yükleyin"
    )
    
    if uploaded_files:
        st.write("Yüklenen görseller:")
        cols = st.columns(min(len(uploaded_files), 3))
        for idx, file in enumerate(uploaded_files):
            cols[idx % 3].image(file, use_column_width=True)
    
    # Oluştur butonu
    if st.button("3D Model Oluştur", disabled=not uploaded_files):
        with st.spinner("3D model oluşturuluyor..."):
            try:
                # Model çıktısını al
                mesh = model.predict(uploaded_files)
                
                # Geçici dosya oluştur
                with tempfile.NamedTemporaryFile(suffix=".obj", delete=False) as tmp:
                    mesh.export(tmp.name)
                    
                    # 3D viewer ile göster
                    st.success("3D model başarıyla oluşturuldu!")
                    st.write("Oluşturulan model:")
                    st.components.v1.iframe(
                        f"https://3dviewer.net/embed.html?model={tmp.name}",
                        height=400
                    )
                    
                    # İndirme butonu
                    with open(tmp.name, "rb") as f:
                        st.download_button(
                            "3D Modeli İndir (OBJ)",
                            f,
                            file_name="model.obj",
                            mime="model/obj"
                        )
                    
                # Geçici dosyayı temizle
                os.unlink(tmp.name)
                
            except Exception as e:
                st.error(f"Model oluşturma hatası: {str(e)}")
                logger.error(f"Model generation error: {str(e)}")

def create_ui(config: Dict[str, Any], text_to_shape: TextToShape = None):
    """Create main UI."""
    # Sayfa yapılandırması
    st.set_page_config(
        page_title="Ötüken3D",
        page_icon="🎨",
        layout="wide"
    )
    
    # Custom CSS
    st.markdown("""
        <style>
        .stApp {
            max-width: 1200px;
            margin: 0 auto;
        }
        .stButton button {
            width: 100%;
            margin-top: 1rem;
        }
        .stProgress .st-bo {
            background-color: #1f77b4;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Sidebar ayarları
    sidebar_config = setup_sidebar()
    
    # Ana içerik
    if sidebar_config["model_type"] == "Metin-to-3D":
        setup_text_to_3d_ui(sidebar_config, text_to_shape)
    else:
        image_to_shape = ImageToShape(device=sidebar_config["device"])
        setup_image_to_3d_ui(sidebar_config, image_to_shape)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "Ötüken3D - Türk kültürüne özgü 3D model üretimi için özelleştirilmiş yapay zeka modeli. "
        "Geliştirici: [GitHub](https://github.com/yourusername)"
    )
