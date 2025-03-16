import yaml


# Cargar configuración desde config.yaml
def load_config(config_path):
    """Carga el archivo de configuración YAML"""
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config
