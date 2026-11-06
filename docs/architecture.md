# System Architecture & Design Document: Problem Statement 1 (ProbID - P1)

## System Overview: Adaptive Retinex–HVI Restoration (ARHR)

The ARHR system is an illumination-adaptive hybrid restoration pipeline designed for low-light imaging challenges: extreme underexposure, sensor noise, non-uniform lighting, and color distortions.

### System Pipeline Architecture

```mermaid
flowchart TD
    In["Input Low-Light Image (I)"] --> Analysis["Stage 1: Image Characteristic Analysis"]
    
    subgraph S1 ["Analysis Metrics"]
        Analysis --> L["Illumination Map L(x,y)"]
        Analysis --> N["Noise Estimate σ_n(x,y)"]
        Analysis --> E["Edge Gradient Map E(x,y)"]
        Analysis --> C["Color & Exposure Statistics"]
    end

    subgraph S2 ["Stage 2: Parallel Restoration"]
        L & N & E & C --> Classical["Classical CV Branch\n• Retinex Decomposition (I = R · L)\n• Guided Illumination Refinement\n• Noise-Adaptive Luminance CLAHE\n• Edge-Preserving Denoising\n• Adaptive Gamma & White Balance"]
        In --> Deep["Deep Learning Branch (HVI-CIDNet)\n• Decoupled Hue/Value/Intensity\n• Illumination-Adaptive Scaling\n• Shadow Detail Recovery\n• Deep Color Restoration"]
        Classical --> Ic["Classical Output (I_classical)"]
        Deep --> Id["Deep Output (I_deep)"]
    end

    subgraph S3 ["Stage 3: Adaptive Hybrid Fusion"]
        In & Ic & Id & L & N & E --> Fusion["Spatial Weight Generator\n• W_o: Dominates in highlights (anti-clipping)\n• W_c: Preserves sharp mid-tone textures\n• W_d: Dominates in dark/noisy shadows\n• Smooth Transition Filter (Guided Blending)"]
        Fusion --> Fused["Fused Image:\nI_hybrid = W_o·I + W_c·I_c + W_d·I_d"]
    end

    subgraph S4 ["Stage 4: Quality-Aware Output Refinement"]
        Fused --> Refine["Highlight Guard & Color Balancing\n• Soft-Knee Tone Mapping\n• Dynamic Clipping Prevention\n• Edge-Aware Selective Sharpening"]
        Refine --> Out["Final Enhanced Image"]
    end

    subgraph S5 ["Stage 5: Evaluation & Deployment"]
        Out --> Eval["Evaluation Suite\n• PSNR / SSIM (Paired Ground Truth)\n• LPIPS (Perceptual Quality)\n• NIQE (No-Reference)\n• Processing Latency (<10s target)\n• Noise Suppression (Wavelet MAD) & Edge Preservation (EPI)"]
        Out --> App["Interactive Application\n• Gradio Web Interface\n• Side-by-side & Slider comparison\n• Intermediate Map Visualization\n• Bad Input & Resolution Guard"]
    end
```

---

## Technical Stages

### 1. Image Characteristic Analysis
* **Illumination Map $L(x,y)$**: Extracted through max-channel approximation or guided filtering in luminance space.
* **Noise Estimation $\sigma_n$**: High-frequency wavelet Median Absolute Deviation (MAD) in flat/shadow regions.
* **Edge Map $E(x,y)$**: Sobel / Scharr gradient magnitude.
* **Exposure Statistics**: Percentage of underexposed ($L < 0.15$) and clipped highlight ($L > 0.95$) pixels.

### 2. Classical Computer Vision Branch
* **Retinex Decomposition**: $I(x,y) = R(x,y) \cdot L(x,y)$.
* **Refinement**: Guided filtering preserves hard edges while smoothing illumination variations, eliminating Retinex halo artifacts.
* **Luminance CLAHE**: Contrast Limited Adaptive Histogram Equalization applied only to luminance ($V$ in HSV or $L^*$ in CIELAB) to avoid color shift.
* **Adaptive Denoising**: Bilateral / guided filtering parameterized inversely to local illumination (stronger denoising where $L$ is low and $\sigma_n$ is high).

### 3. Deep Learning Branch (HVI-CIDNet)
* Operates on decoupled representations to recover clean structural information without amplifying color noise.
* Utilizes pretrained checkpoints optimized on benchmark low-light datasets (LOL / LOL-v2).

### 4. Adaptive Spatial Fusion
* Spatially varying blending weights:
  $$I_{\text{hybrid}}(x,y) = W_o(x,y)I(x,y) + W_c(x,y)I_{\text{classical}}(x,y) + W_d(x,y)I_{\text{deep}}(x,y)$$
  subject to $W_o(x,y) + W_c(x,y) + W_d(x,y) = 1$.
* Avoids overexposure by giving high weight to $W_o$ in bright regions, avoiding deep learning artifacts in bright areas and classical noise amplification in deep shadows.

### 5. Quality-Aware Output Refinement
* Highlight guard preventing clipped channels ($>0.98$ intensity).
* Soft-knee tone mapping and conservative sharpening applied only in confident structural regions.
