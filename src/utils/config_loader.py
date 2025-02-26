import yaml


# Cargar configuración desde config.yaml
def load_ner_config(config_path="config.yaml"):
    """Carga el archivo de configuración YAML"""
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config
