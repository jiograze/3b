"""
API Rotaları
"""

import tempfile
import os
from pathlib import Path
from typing import List, Dict, Any
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel

from src.processing.mesh_processor import MeshProcessor
from src.processing.texture_processor import TextureProcessor
from src.processing.format_converter import FormatConverter

router = APIRouter()

# API modelleri
class ProcessingResponse(BaseModel):
    """İşlem yanıtı"""
    success: bool
    message: str
    data: Dict[str, Any] = {}
    
class BatchResponse(BaseModel):
    """Toplu işlem yanıtı"""
    successful: List[str]
    failed: List[Dict[str, str]]

def cleanup_temp_files(*files):
    """Geçici dosyaları temizler"""
    for file in files:
        try:
            if file and os.path.exists(file):
                os.unlink(file)
        except Exception:
            pass
    
@router.post("/mesh/optimize", response_model=ProcessingResponse)
async def optimize_mesh(
    file: UploadFile = File(...),
    target_faces: int = Form(None)
):
    """3B modeli optimize eder
    
    Args:
        file: Model dosyası
        target_faces: Hedef üçgen sayısı
    """
    temp_in = None
    temp_out = None
    
    try:
        processor = MeshProcessor()
        
        # Geçici dosyaları oluştur
        temp_in = tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix).name
        temp_out = tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix).name
        
        # Yüklenen dosyayı kaydet
        content = await file.read()
        with open(temp_in, 'wb') as f:
            f.write(content)
        
        # Optimize et
        processor.optimize(
            temp_in,
            temp_out,
            target_faces=target_faces
        )
        
        # Sonucu oku
        with open(temp_out, 'rb') as f:
            result = f.read()
            
        return ProcessingResponse(
            success=True,
            message="Model başarıyla optimize edildi",
            data={'content': result}
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
        
    finally:
        # Geçici dosyaları temizle
        cleanup_temp_files(temp_in, temp_out)

@router.post("/mesh/repair", response_model=ProcessingResponse)
async def repair_mesh(
    file: UploadFile = File(...),
    fix_normals: bool = Form(True),
    remove_duplicates: bool = Form(True)
):
    """3B modeldeki hataları onarır
    
    Args:
        file: Model dosyası
        fix_normals: Normaller düzeltilsin mi?
        remove_duplicates: Tekrarlanan noktalar silinsin mi?
    """
    temp_in = None
    temp_out = None
    
    try:
        processor = MeshProcessor()
        
        # Geçici dosyaları oluştur
        temp_in = tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix).name
        temp_out = tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix).name
        
        # Yüklenen dosyayı kaydet
        content = await file.read()
        with open(temp_in, 'wb') as f:
            f.write(content)
        
        # Onar
        processor.repair(
            temp_in,
            temp_out,
            fix_normals=fix_normals,
            remove_duplicates=remove_duplicates
        )
        
        # Sonucu oku
        with open(temp_out, 'rb') as f:
            result = f.read()
            
        return ProcessingResponse(
            success=True,
            message="Model başarıyla onarıldı",
            data={'content': result}
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
        
    finally:
        # Geçici dosyaları temizle
        cleanup_temp_files(temp_in, temp_out)

@router.post("/texture/process", response_model=ProcessingResponse)
async def process_texture(
    file: UploadFile = File(...),
    width: int = Form(None),
    height: int = Form(None),
    quality: int = Form(None)
):
    """Doku haritasını işler
    
    Args:
        file: Doku dosyası
        width: Yeni genişlik
        height: Yeni yükseklik
        quality: Çıktı kalitesi (0-100)
    """
    temp_in = None
    temp_out = None
    
    try:
        processor = TextureProcessor()
        
        # Geçici dosyaları oluştur
        temp_in = tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix).name
        temp_out = tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix).name
        
        # Yüklenen dosyayı kaydet
        content = await file.read()
        with open(temp_in, 'wb') as f:
            f.write(content)
        
        # İşle
        resize = (width, height) if width and height else None
        processor.process(
            temp_in,
            temp_out,
            resize=resize,
            quality=quality
        )
        
        # Sonucu oku
        with open(temp_out, 'rb') as f:
            result = f.read()
            
        return ProcessingResponse(
            success=True,
            message="Doku başarıyla işlendi",
            data={'content': result}
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
        
    finally:
        # Geçici dosyaları temizle
        cleanup_temp_files(temp_in, temp_out)

@router.post("/convert", response_model=ProcessingResponse)
async def convert_format(
    file: UploadFile = File(...),
    output_format: str = Form(...),
    preserve_materials: bool = Form(True),
    optimize_mesh: bool = Form(True)
):
    """3B model formatını dönüştürür
    
    Args:
        file: Model dosyası
        output_format: Çıktı formatı
        preserve_materials: Materyaller korunsun mu?
        optimize_mesh: Mesh optimize edilsin mi?
    """
    temp_in = None
    temp_out = None
    
    try:
        converter = FormatConverter()
        
        # Çıktı formatını normalize et
        if not output_format.startswith('.'):
            output_format = f".{output_format}"
            
        # Geçici dosyaları oluştur
        temp_in = tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix).name
        temp_out = tempfile.NamedTemporaryFile(delete=False, suffix=output_format).name
        
        # Yüklenen dosyayı kaydet
        content = await file.read()
        with open(temp_in, 'wb') as f:
            f.write(content)
        
        # Dönüştür
        converter.convert(
            temp_in,
            temp_out,
            preserve_materials=preserve_materials,
            optimize_mesh=optimize_mesh
        )
        
        # Sonucu oku
        with open(temp_out, 'rb') as f:
            result = f.read()
            
        return ProcessingResponse(
            success=True,
            message="Model başarıyla dönüştürüldü",
            data={'content': result}
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
        
    finally:
        # Geçici dosyaları temizle
        cleanup_temp_files(temp_in, temp_out)

@router.post("/batch/convert", response_model=BatchResponse)
async def batch_convert(
    files: List[UploadFile],
    output_format: str = Form(...),
    preserve_materials: bool = Form(True),
    optimize_mesh: bool = Form(True)
):
    """Birden fazla 3B modeli dönüştürür
    
    Args:
        files: Model dosyaları
        output_format: Çıktı formatı
        preserve_materials: Materyaller korunsun mu?
        optimize_mesh: Mesh optimize edilsin mi?
    """
    temp_files = []
    
    try:
        converter = FormatConverter()
        
        # Geçici klasörleri oluştur
        with tempfile.TemporaryDirectory() as temp_in_dir, \
             tempfile.TemporaryDirectory() as temp_out_dir:
            # Dosyaları kaydet
            for file in files:
                input_path = Path(temp_in_dir) / file.filename
                content = await file.read()
                with open(input_path, 'wb') as f:
                    f.write(content)
                temp_files.append(str(input_path))
                    
            # Toplu dönüştür
            results = converter.batch_convert(
                temp_in_dir,
                temp_out_dir,
                output_format=output_format,
                preserve_materials=preserve_materials,
                optimize_mesh=optimize_mesh
            )
            
        return BatchResponse(
            successful=results['successful'],
            failed=[{'path': f['path'], 'error': f['error']} for f in results['failed']]
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
        
    finally:
        # Geçici dosyaları temizle
        cleanup_temp_files(*temp_files)

@router.get("/formats")
async def get_formats():
    """Desteklenen dosya formatlarını listeler"""
    try:
        converter = FormatConverter()
        formats = {
            '3d': sorted(list(converter.SUPPORTED_FORMATS['3d'])),
            'texture': sorted(list(converter.SUPPORTED_FORMATS['texture']))
        }
        return ProcessingResponse(
            success=True,
            message="Desteklenen formatlar listelendi",
            data={'formats': formats}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 