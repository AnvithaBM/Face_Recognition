#!/usr/bin/env python3
"""
Validation script for the face recognition notebook.
Checks notebook structure, syntax, and key components without executing cells.
"""

import json
import sys

def validate_notebook(notebook_path):
    """Validate the notebook structure and components."""
    
    print("="*70)
    print("NOTEBOOK VALIDATION REPORT")
    print("="*70)
    
    try:
        with open(notebook_path, 'r') as f:
            notebook = json.load(f)
    except FileNotFoundError:
        print(f"✗ ERROR: Notebook not found at {notebook_path}")
        return False
    except json.JSONDecodeError as e:
        print(f"✗ ERROR: Invalid JSON in notebook: {e}")
        return False
    
    # Basic structure validation
    print("\n1. STRUCTURE VALIDATION")
    print("-" * 70)
    
    if 'cells' not in notebook:
        print("✗ ERROR: No 'cells' field in notebook")
        return False
    
    total_cells = len(notebook['cells'])
    code_cells = sum(1 for c in notebook['cells'] if c['cell_type'] == 'code')
    markdown_cells = sum(1 for c in notebook['cells'] if c['cell_type'] == 'markdown')
    
    print(f"✓ Total cells: {total_cells}")
    print(f"✓ Code cells: {code_cells}")
    print(f"✓ Markdown cells: {markdown_cells}")
    
    if 'metadata' in notebook:
        kernel = notebook['metadata'].get('kernelspec', {}).get('display_name', 'Unknown')
        print(f"✓ Kernel: {kernel}")
    
    # Check for required sections
    print("\n2. REQUIRED SECTIONS")
    print("-" * 70)
    
    required_sections = {
        'imports': ('import', 'numpy'),
        'data_loading': ('load_hyperspectral_image', 'def'),
        'gabor_transform': ('gabor', 'getgaborkernel'),
        'model_architecture': ('mobilenet', 'vgg16'),
        'model_building': ('build_face_recognition_model', 'def'),
        'model_compilation': ('model.compile', 'optimizer'),
        'training': ('model.fit', 'history'),
        'evaluation': ('classification_report', 'confusion_matrix')
    }
    
    found_sections = {key: False for key in required_sections}
    
    for cell in notebook['cells']:
        if cell['cell_type'] != 'code':
            continue
        source = ''.join(cell['source']).lower()
        
        for section, keywords in required_sections.items():
            if all(keyword.lower() in source for keyword in keywords):
                found_sections[section] = True
    
    all_found = True
    for section, found in found_sections.items():
        status = "✓" if found else "✗"
        section_name = section.replace('_', ' ').title()
        print(f"{status} {section_name}")
        if not found:
            all_found = False
    
    # Check for syntax errors in code cells
    print("\n3. SYNTAX VALIDATION")
    print("-" * 70)
    
    syntax_errors = []
    for i, cell in enumerate(notebook['cells'], 1):
        if cell['cell_type'] != 'code':
            continue
        
        source = ''.join(cell['source'])
        if not source.strip():
            continue
        
        # Remove Jupyter magic commands before checking syntax
        lines = source.split('\n')
        filtered_lines = [line for line in lines if not line.strip().startswith('%')]
        filtered_source = '\n'.join(filtered_lines)
        
        if not filtered_source.strip():
            continue
        
        try:
            compile(filtered_source, f'<cell {i}>', 'exec')
        except SyntaxError as e:
            syntax_errors.append((i, str(e)))
    
    if syntax_errors:
        print(f"✗ Found {len(syntax_errors)} syntax errors:")
        for cell_num, error in syntax_errors[:5]:  # Show first 5
            print(f"  Cell {cell_num}: {error}")
        all_found = False
    else:
        print(f"✓ No syntax errors found in {code_cells} code cells")
    
    # Check for key model features
    print("\n4. MODEL ARCHITECTURE VALIDATION")
    print("-" * 70)
    
    model_features = {
        'transfer_learning': False,
        'dual_branch': False,
        'mobilenet_support': False,
        'vgg16_support': False,
        'channel_splitting': False
    }
    
    for cell in notebook['cells']:
        source = ''.join(cell['source']).lower()
        
        if 'mobilenet' in source and 'from keras.applications' in source:
            model_features['mobilenet_support'] = True
        if 'vgg16' in source and 'from keras.applications' in source:
            model_features['vgg16_support'] = True
        if 'lambda' in source and 'x[:, :, :, :3]' in source:
            model_features['channel_splitting'] = True
        if 'base_model' in source and 'weights=' in source:
            model_features['transfer_learning'] = True
        if 'concatenate' in source and 'rgb' in source and 'gabor' in source:
            model_features['dual_branch'] = True
    
    for feature, found in model_features.items():
        status = "✓" if found else "✗"
        feature_name = feature.replace('_', ' ').title()
        print(f"{status} {feature_name}")
        if not found:
            all_found = False
    
    # Summary
    print("\n" + "="*70)
    if all_found:
        print("✓ VALIDATION PASSED: All requirements met!")
        print("="*70)
        return True
    else:
        print("⚠ VALIDATION WARNING: Some checks failed (see above)")
        print("="*70)
        return False

if __name__ == '__main__':
    notebook_path = 'face_recognition_hyperspectral (3).ipynb'
    success = validate_notebook(notebook_path)
    sys.exit(0 if success else 1)
