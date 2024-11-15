from cx_Freeze import setup, Executable

# Lista de arquivos adicionais
files = [
    "config.json",
    ("processing/background_removal.py", "processing"),
    ("processing/u2net.py", "processing"),
    ("U2Net", "U2Net")
]

# Configuração do executável
setup(
    name="AppPronto",
    version="1.0",
    description="Editor de Imagens",
    options={"build_exe": {"include_files": files}},
    executables=[Executable("main.py", base="Win32GUI")]  # Remova base="Win32GUI" se precisar de console
)
