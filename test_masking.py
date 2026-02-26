import torch
from src.model.CLIP import DEC_CLIP, DEC_CLIPConfig

def run_sanity_check():
    print("1. Initializing Config...")
    config = DEC_CLIPConfig(
        vision_encoder="dcformer", # or vit3d
        language_model_name_or_path="/Med3DVLM/src/model/Med3DVLM-Qwen-2.5-7B", # Lightweight text encoder for testing
        input_size=(128, 256, 256), # Adjust to your actual D, H, W
        use_masking=True,
        mask_ratio=0.5 # Prune 50% of tokens
    )

    print("2. Building Model...")
    model = DEC_CLIP(config)
    model = model.cuda()
    model.eval()

    print("3. Generating Dummy Data...")
    # Shape: [Batch, Channels, Depth, Height, Width]
    dummy_images = torch.randn(2, 1, 128, 256, 256).cuda()
    dummy_input_ids = torch.randint(0, 1000, (2, 32)).cuda()
    dummy_attention_mask = torch.ones(2, 32).cuda()
    dummy_labels = torch.arange(2).cuda()

    print("4. Testing Forward Pass...")
    outputs = model(
        images=dummy_images, 
        input_ids=dummy_input_ids, 
        attention_mask=dummy_attention_mask, 
        labels=dummy_labels
    )
    print(f"Success! Total Loss: {outputs['loss'].item():.4f}")

    print("5. Extracting Saliency Mask...")
    mask_scores = model.visualize_mask(dummy_images)
    print(f"Mask Scores Shape: {mask_scores.shape}") # Should be [2, N] (where N is sequence length)
    
    # Check score distribution
    print(f"Mean score: {mask_scores.mean().item():.4f} (Should start around 0.5 before training)")

if __name__ == "__main__":
    run_sanity_check()