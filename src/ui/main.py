"""
Ana UI modülü
"""

import streamlit as st
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import tempfile
import os
import shutil

from src.core.exceptions import ModelError, ProcessingError
from src.core.types import PathLike
from src.model3d_integration import TextToShape, ImageToShape
from src.processing import FormatConverter

logger = logging.getLogger(__name__)

class UI:
    """Ana UI sınıfı"""
    
    SUPPORTED_FORMATS = ['.obj', '.stl', '.ply', '.glb', '.gltf', '.fbx', '.dae']
    MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
    
    def __init__(self):
        """UI başlatıcı"""
        self.temp_dir = None
        self._setup_temp_dir()
        self._init_session_state()
        
    def _setup_temp_dir(self):
        """Geçici dizin oluşturur"""
        try:
            self.temp_dir = tempfile.mkdtemp(prefix="otuken3d_ui_")
        except Exception as e:
            logger.error(f"Geçici dizin oluşturma hatası: {str(e)}")
            raise RuntimeError("Uygulama başlatılamadı")
            
    def _init_session_state(self):
        """Oturum durumunu başlatır"""
        if 'history' not in st.session_state:
            st.session_state.history = []
            
    def _validate_file(self, file) -> None:
        """Dosya doğrulama
        
        Args:
            file: Yüklenen dosya
            
        Raises:
            ValueError: Geçersiz dosya
        """
        if not file:
            raise ValueError("Dosya seçilmedi")
            
        if file.size > self.MAX_FILE_SIZE:
            raise ValueError(f"Dosya boyutu çok büyük (max: {self.MAX_FILE_SIZE/1024/1024:.1f}MB)")
            
        file_ext = Path(file.name).suffix.lower()
        if file_ext not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Desteklenmeyen dosya formatı: {file_ext}")
            
    def _save_upload(self, file) -> Path:
        """Yüklenen dosyayı kaydeder
        
        Args:
            file: Yüklenen dosya
            
        Returns:
            Dosya yolu
            
        Raises:
            RuntimeError: Kaydetme hatası
        """
        try:
            save_path = Path(self.temp_dir) / file.name
            with open(save_path, 'wb') as f:
                f.write(file.getbuffer())
            return save_path
        except Exception as e:
            logger.error(f"Dosya kaydetme hatası: {str(e)}")
            raise RuntimeError(f"Dosya kaydedilemedi: {str(e)}")
            
    def run(self):
        """UI'ı çalıştırır"""
        try:
            st.title("Otuken3D")
            st.write("3B Model İşleme ve Dönüştürme Aracı")
            
            # Ana sekmeler
            tab1, tab2, tab3 = st.tabs(["Metin → Model", "Görüntü → Model", "Format Dönüştürme"])
            
            with tab1:
                self._text_to_shape_ui()
                
            with tab2:
                self._image_to_shape_ui()
                
            with tab3:
                self._format_converter_ui()
                
            # İşlem geçmişi
            if st.session_state.history:
                st.subheader("İşlem Geçmişi")
                for item in reversed(st.session_state.history):
                    st.write(item)
                    
        except Exception as e:
            logger.error(f"UI hatası: {str(e)}")
            st.error(f"Beklenmeyen bir hata oluştu: {str(e)}")
            
    def _text_to_shape_ui(self):
        """Metin → Model UI'ı"""
        st.header("Metinden 3B Model")
        
        try:
            # Kullanıcı girdisi
            text = st.text_area("Model açıklaması", 
                              help="3B modeli tanımlayan metni girin")
                              
            if st.button("Model Oluştur", disabled=not text):
                with st.spinner("Model oluşturuluyor..."):
                    try:
                        # Model oluşturma
                        converter = TextToShape(model_path="models/text2shape")
                        result = converter.process_text(text)
                        
                        # Sonucu göster
                        st.success("Model başarıyla oluşturuldu!")
                        st.write(result)
                        
                        # Geçmişe ekle
                        st.session_state.history.append({
                            'type': 'text_to_shape',
                            'input': text,
                            'output': result
                        })
                        
                    except (ModelError, ProcessingError) as e:
                        st.error(f"Model oluşturma hatası: {str(e)}")
                        logger.error(f"Model oluşturma hatası: {str(e)}")
                        
        except Exception as e:
            logger.error(f"Text-to-shape UI hatası: {str(e)}")
            st.error(f"Beklenmeyen bir hata oluştu: {str(e)}")
            
    def _image_to_shape_ui(self):
        """Görüntü → Model UI'ı"""
        st.header("Görüntüden 3B Model")
        
        try:
            # Dosya yükleme
            file = st.file_uploader("Görüntü seç", 
                                  type=['png', 'jpg', 'jpeg'],
                                  help="PNG veya JPEG formatında görüntü")
                                  
            if file:
                try:
                    # Dosya doğrulama
                    if file.size > self.MAX_FILE_SIZE:
                        st.error(f"Dosya boyutu çok büyük (max: {self.MAX_FILE_SIZE/1024/1024:.1f}MB)")
                        return
                        
                    # Görüntüyü göster
                    st.image(file, caption="Yüklenen görüntü")
                    
                    if st.button("Model Oluştur"):
                        with st.spinner("Model oluşturuluyor..."):
                            try:
                                # Dosyayı kaydet
                                image_path = self._save_upload(file)
                                
                                # Model oluştur
                                converter = ImageToShape(model_path="models/image2shape")
                                result = converter.process_image(image_path)
                                
                                # Sonucu göster
                                st.success("Model başarıyla oluşturuldu!")
                                st.write(result)
                                
                                # Geçmişe ekle
                                st.session_state.history.append({
                                    'type': 'image_to_shape',
                                    'input': file.name,
                                    'output': result
                                })
                                
                            except (ModelError, ProcessingError) as e:
                                st.error(f"Model oluşturma hatası: {str(e)}")
                                logger.error(f"Model oluşturma hatası: {str(e)}")
                                
                except Exception as e:
                    st.error(f"Dosya işleme hatası: {str(e)}")
                    logger.error(f"Dosya işleme hatası: {str(e)}")
                    
        except Exception as e:
            logger.error(f"Image-to-shape UI hatası: {str(e)}")
            st.error(f"Beklenmeyen bir hata oluştu: {str(e)}")
            
    def _format_converter_ui(self):
        """Format dönüştürme UI'ı"""
        st.header("Format Dönüştürme")
        
        try:
            # Dosya yükleme
            file = st.file_uploader("3B model seç",
                                  type=[fmt[1:] for fmt in self.SUPPORTED_FORMATS],
                                  help="Desteklenen formatlar: " + ", ".join(self.SUPPORTED_FORMATS))
                                  
            if file:
                try:
                    # Dosya doğrulama
                    self._validate_file(file)
                    
                    # Format seçimi
                    input_format = Path(file.name).suffix.lower()
                    converter = FormatConverter()
                    available_formats = [fmt for fmt in self.SUPPORTED_FORMATS 
                                      if fmt in converter.CONVERSION_MATRIX[input_format]]
                                      
                    output_format = st.selectbox("Hedef format",
                                               available_formats,
                                               help="Dönüştürülecek format")
                                               
                    if st.button("Dönüştür"):
                        with st.spinner("Dönüştürülüyor..."):
                            try:
                                # Dosyayı kaydet
                                input_path = self._save_upload(file)
                                
                                # Dönüştür
                                result = converter.convert(input_path, output_format)
                                
                                # Sonucu göster
                                st.success("Dönüştürme başarılı!")
                                st.write(result)
                                
                                # Geçmişe ekle
                                st.session_state.history.append({
                                    'type': 'format_converter',
                                    'input': file.name,
                                    'output': result
                                })
                                
                            except (ValueError, ProcessingError) as e:
                                st.error(f"Dönüştürme hatası: {str(e)}")
                                logger.error(f"Dönüştürme hatası: {str(e)}")
                                
                except ValueError as e:
                    st.error(str(e))
                except Exception as e:
                    st.error(f"Dosya işleme hatası: {str(e)}")
                    logger.error(f"Dosya işleme hatası: {str(e)}")
                    
        except Exception as e:
            logger.error(f"Format converter UI hatası: {str(e)}")
            st.error(f"Beklenmeyen bir hata oluştu: {str(e)}")
            
    def __del__(self):
        """Yıkıcı metod - kaynakları temizler"""
        if self.temp_dir and os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
            except Exception as e:
                logger.warning(f"Geçici dizin temizleme hatası: {str(e)}")
                
def main():
    """Ana fonksiyon"""
    try:
        ui = UI()
        ui.run()
    except Exception as e:
        logger.error(f"Uygulama hatası: {str(e)}")
        st.error("Uygulama başlatılamadı. Lütfen daha sonra tekrar deneyin.")
        
if __name__ == "__main__":
    main()
