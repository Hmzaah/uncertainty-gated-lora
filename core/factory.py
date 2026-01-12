import yaml
import torch
from transformers import AutoImageProcessor, ResNetForImageClassification
from peft import LoraConfig, get_peft_model, TaskType

class AdapterFactory:
    @staticmethod
    def load_config(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    @staticmethod
    def create_model(config_path):
        """
        Builds the model + adapter based on YAML instructions.
        """
        cfg = AdapterFactory.load_config(config_path)
        model_id = cfg['model']['base_model']
        
        print(f"🏭 [Factory] Loading Base Model: {model_id}")
        
        # 1. Load Backbone
        # We use 'ignore_mismatched_sizes' to handle custom class counts if needed
        model = ResNetForImageClassification.from_pretrained(
            model_id,
            num_labels=cfg['model']['num_classes'],
            ignore_mismatched_sizes=True
        )

        # 2. Configure Adapter (The "Smart" Part)
        method = cfg['adapter']['method'].lower()
        use_dora = (method == "dora")
        
        print(f"🔧 [Factory] Injecting Adapter -> Type: {method.upper()} | Rank: {cfg['adapter']['r']}")

        peft_config = LoraConfig(
            inference_mode=False,
            r=cfg['adapter']['r'],
            lora_alpha=cfg['adapter']['lora_alpha'],
            lora_dropout=cfg['adapter']['lora_dropout'],
            target_modules=cfg['adapter']['target_modules'],
            use_dora=use_dora  # This toggle enables Weight-Decomposed LoRA
        )

        # 3. Inject
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        
        return model, cfg

if __name__ == "__main__":
    # Quick Smoke Test
    model, cfg = AdapterFactory.create_model("configs/master_template.yaml")
