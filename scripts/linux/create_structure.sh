#!/bin/bash

# Create main project structure
mkdir -p Ötüken3D/{data,models,modules,utils,tests,docs,scripts}

# Create data subdirectories
mkdir -p Ötüken3D/data/{images,3d_models,text_prompts,datasets/{COCO,ImageNet,ShapeNet,Pix3D}}

# Create module subdirectories
mkdir -p Ötüken3D/modules/{data_management,nlp,image_processing,model_generation,training,evaluation,ui,security,deployment}

# Create utils subdirectories
mkdir -p Ötüken3D/utils/{helpers,config,logging}

# Create test subdirectories
mkdir -p Ötüken3D/tests/{unit,integration,end_to_end}

# Create docs subdirectories
mkdir -p Ötüken3D/docs/{user_guide,api_docs,developer_guide}

# Create necessary empty files
touch Ötüken3D/README.md
touch Ötüken3D/requirements.txt
touch Ötüken3D/setup.py
touch Ötüken3D/app.py

# Create module initialization files
find Ötüken3D/modules -type d -exec touch {}/__init__.py \;
