# PosFormer Integration - Final Solution

## Key Findings

After extensive testing of the PosFormer model with Docker, we have identified and resolved the key issues with processing the PNG images:

1. **Model Loading**: The model loads correctly from the checkpoint at `PosFormer-main/lightning_logs/version_0/checkpoints/best.ckpt`.

2. **Input Format Mismatch**: The critical issue was a mismatch between the expected input format and the format of your images:
   - **CROHME dataset format**: White strokes on black background
   - **Supernote images format**: Black strokes on white background

3. **Solution**: Setting `invert=True` when preprocessing Supernote images resolves the issue by converting them to the CROHME format.

## Correct Implementation

### 1. Proper Tensor Dimensions

The model expects input with these dimensions:
- Image tensor: `[batch_size=1, channels=1, height=256, width=1024]`
- Mask tensor: `[batch_size=1, height=256, width=1024]`

### 2. Correct Preprocessing Pipeline

```python
# Load the image
img = Image.open(image_path).convert('L')  # Convert to grayscale
img_array = np.array(img)

# Preprocess WITH inversion for Supernote images
processed = preprocess_image(
    image=img_array,
    target_resolution=(256, 1024),
    invert=True,  # Critical for Supernote images
    normalize=False
)

# Fix tensor dimensions
processed_np = processed.squeeze()  # Remove extra dimensions
tensor = torch.from_numpy(processed_np).float() / 255.0
tensor = tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims

# Create attention mask
mask = torch.zeros((1, 256, 1024), dtype=torch.bool, device='cpu')
```

### 3. Model Inference

```python
# Run encoder
with torch.no_grad():
    feature, enc_mask = model.encoder(tensor, mask)

# Run beam search
with torch.no_grad():
    hypotheses = model.beam_search(
        tensor,
        mask,
        beam_size=8,
        max_len=50,
        alpha=1.0,
        early_stopping=True,
        temperature=1.0
    )
    
    # Get the LaTeX from the best hypothesis
    if hypotheses:
        best_hyp = hypotheses[0]
        latex = vocab.indices2label(best_hyp.seq)
```

## Docker Command

The final Docker command for running PosFormer with your images:

```bash
docker run --rm \
  -v "/path/to/PosFormer-main:/posformer" \
  -v "/path/to/supernote-math:/app" \
  -e "OMP_NUM_THREADS=1" \
  -e "OPENBLAS_NUM_THREADS=1" \
  -e "MKL_NUM_THREADS=1" \
  supernote-posformer python /app/your_script.py
```

## Performance Considerations

1. **CPU vs. GPU**: The model runs on CPU but would be much faster with GPU acceleration.
2. **Threading**: Setting `OMP_NUM_THREADS=1` and other thread limiting variables prevents OpenBLAS hanging issues.
3. **Beam Search Parameters**: For faster processing, consider reducing `beam_size` (e.g., to 5) and `max_len` (e.g., to 30).

## Known Limitations

1. **Recognition Quality**: Even with the correct format, the model's output may not perfectly match your handwriting due to training data differences.
2. **Processing Time**: On CPU, expect approximately 30-40 seconds per image with beam search.
3. **Memory Usage**: The model requires approximately 1-2GB of RAM.

## Next Steps

1. **GPU Acceleration**: Consider using a GPU for faster inference.
2. **Fine-tuning**: The model could be fine-tuned on your specific handwriting style.
3. **Optimized Parameters**: Further tuning of beam search parameters could improve results.

## Conclusion

The PosFormer model is successfully loading and processing images in Docker. The critical insight was understanding the format difference between CROHME dataset (white-on-black) and Supernote images (black-on-white). With the `invert=True` parameter properly set, the model can process Supernote handwritten math expressions.