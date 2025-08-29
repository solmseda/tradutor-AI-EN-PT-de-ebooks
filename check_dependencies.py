#!/usr/bin/env python3

def check_dependencies():
    dependencies = [
        ('torch', 'pip install torch --index-url https://download.pytorch.org/whl/cpu'),
        ('transformers', 'pip install transformers'),
        ('sentencepiece', 'pip install sentencepiece'),
        ('ebooklib', 'pip install ebooklib'),
        ('bs4', 'pip install beautifulsoup4'),
        ('PyQt5', 'pip install pyqt5'),
    ]
    
    print("Verificando dependências...")
    all_ok = True
    
    for package, install_cmd in dependencies:
        try:
            __import__(package)
            print(f"✅ {package} - OK")
        except ImportError:
            print(f"❌ {package} - FALTANDO")
            print(f"   Execute: {install_cmd}")
            all_ok = False
    
    if all_ok:
        print("\n🎉 Todas as dependências estão instaladas!")
    else:
        print("\n⚠️  Instale as dependências faltantes e execute novamente")

if __name__ == "__main__":
    check_dependencies()