import streamlit as st
import torch
from modules.ui.main import create_ui
from modules.core.config import load_config
from modules.model3d_integration.text_to_shape import TextToShape

def main():
    st.set_page_config(
        page_title="Ötüken3D",
        page_icon="🎨",
        layout="wide"
    )
    
    # GPU kontrolü - AMD için özel kontrol
    device = torch.device("cpu")
    if torch.cuda.is_available():
        try:
            # GPU'yu test et
            test_tensor = torch.tensor([1.0]).cuda()
            device = torch.device("cuda")
            st.sidebar.success(f"GPU Bulundu: {torch.cuda.get_device_name(0)}")
        except Exception as e:
            st.sidebar.warning(f"GPU bulundu fakat kullanılamıyor: {str(e)}")
            device = torch.device("cpu")
    else:
        st.sidebar.info("CPU kullanılıyor")
    
    config = load_config()
    text_to_shape = TextToShape(device=device)
    
    # UI oluştur
    create_ui(config, text_to_shape)

if __name__ == "__main__":
    main()