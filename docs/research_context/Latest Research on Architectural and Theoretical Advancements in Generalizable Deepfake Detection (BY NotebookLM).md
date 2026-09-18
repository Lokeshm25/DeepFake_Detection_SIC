Architectural and Theoretical Advancements in Generalizable Deepfake Detection
Systematic Evaluation of the Baseline Capstone Pipeline and the Gemini-ChatGPT Multi-Branch Research Decision
Analysis of the Baseline Capstone Pipeline
The baseline deepfake detection pipeline, commonly deployed in academic capstone projects, relies on a sequential frame-by-frame or window-based video analysis architecture [cite: 1]. The pipeline processes input video files through a series of discrete spatial and temporal operations designed to extract and normalize face regions before binary classification [cite: 1, 2].
[Input Video] ➔ [Frame Decoding (OpenCV/FFmpeg)] ➔ [Face Detection & Alignment (MTCNN/dlib)] 
                     │
                     ▼
[Sequential Stacking (T=20)] ➔ [Inference Engine (EfficientNet/Xception)] ➔ [Class Logits]
First, video streams are decoded using computer vision libraries such as OpenCV or FFmpeg, sampling frames at a fixed rate (e.g., five frames per second) to reduce temporal redundancy and minimize computational overhead [cite: 1]. Second, face detection and landmark localization are performed on each decoded frame [cite: 1]. Standard implementations utilize Multi-Task Convolutional Neural Networks (MTCNN) or dlib to return bounding boxes and five core facial landmarks [cite: 1, 3].
The face is cropped and aligned using these landmarks [cite: 1, 3]. To preserve contextual forensic cues around the facial boundary—such as blending edges, chin contours, and hair-line interfaces—a margin (typically 30%) is added to the bounding box dimensions before resizing the crop to standard network inputs, such as 224×224 or 299×299 pixels [cite: 1, 3].
Third, these normalized face crops are structured into temporal sequences of fixed length T (e.g., T=20 frames) [cite: 1]. If a video segment lacks sufficient frames, sequences are padded or looped; if it exceeds the limit, multiple sequential overlapping windows are generated [cite: 1]. Finally, these tensor sequences are passed into a deep feature extractor, such as Xception, ResNet, or EfficientNet, where spatial features are pooled across the temporal axis to yield binary classification logits representing the authenticity score [cite: 1, 4, 5].
Evaluation of the Gemini-ChatGPT Research Decision Components
The research decision reached by the Gemini and ChatGPT models proposes a multi-branch architecture integrating four distinct methodologies: Frequency in Face Forgery Network (F3-Net) [cite: 6], Self-Blended Images (SBI) [cite: 7, 8], Remote Photoplethysmography (rPPG) [cite: 9, 10], and the Interpretable Spatial-Temporal Video Transformer (ISTVT) [cite: 11, 12].
Frequency in Face Forgery Network (F3-Net)
F3-Net is designed to address the limitation of spatial CNNs, which often overlook high-frequency anomalies introduced during the up-sampling stages of generative pipelines [cite: 4, 13]. F3-Net processes the input image through two specialized branches: Frequency-Aware Decomposition (FAD) and Local Frequency Statistics (LFS) [cite: 6, 14]. FAD dynamically decomposes the input image I∈R 
H×W×3
  into low, mid, and high-frequency components using Discrete Fourier Transform (DFT) filters [cite: 6, 15]. Mathematically, the 2D DFT of the image channel is expressed as [cite: 15]:
F(u,v)= 
x=0
∑
H−1
​
  
y=0
∑
W−1
​
 I(x,y)e 
−j2π( 
H
ux
​
 + 
W
vy
​
 )
 
The frequency spectrum is partitioned into three bands using binary frequency-band selectors H 
low
​
 , H 
mid
​
 , and H 
high
​
  [cite: 15]:
F 
band
​
 (u,v)=F(u,v)⊙H 
band
​
 (u,v)
An Inverse DFT (iDFT) projects these filtered representations back to the spatial domain, allowing a standard CNN backbone to learn band-specific anomalies [cite: 13, 15]:
I 
band
​
 (x,y)=F 
−1
 {F 
band
​
 (u,v)}
The LFS branch operates locally to capture the statistical distribution of frequency components [cite: 4, 16]. It utilizes a sliding window to compute block-wise 2D Discrete Cosine Transforms (DCT), extracting statistical descriptors (such as mean, variance, and energy) over local frequency coefficients to expose localized manipulation footprints [cite: 4, 6, 16]. F3-Net supports four operational modes [cite: 6]:
'FAD': Evaluates only the frequency-decomposed spatial representations [cite: 6].
'LFS': Focuses exclusively on the localized block-wise frequency statistics [cite: 6].
'Both': Concatenates the output vectors of both branches before classification [cite: 6].
'Mix': Employs a cross-attention mechanism to adaptively combine FAD and LFS features [cite: 6].
Self-Blended Images (SBI)
SBI is a data synthesis strategy that addresses generalization failures on unseen generative architectures [cite: 7, 8]. Rather than training on fakes produced by specific GANs or diffusion models, SBI generates pseudo-forgeries using a single real image [cite: 7, 8]. This forces the detector to focus on generic blending boundaries, which are common to most face-swap and face-reenactment pipelines [cite: 7, 8, 17].
During training, a real face image I 
real
​
  is selected [cite: 8]. A target face area is cropped using landmark coordinates and subjected to random geometric and photometric transformations (such as scaling, rotation, elastic deformation, and color jittering) to construct a source patch I 
src
​
  [cite: 8, 18]. A continuous, smoothed blending mask M∈[0,1] 
H×W
  is generated by applying morphological erosion and Gaussian blur to the facial landmark hull [cite: 8, 19]. The pseudo-fake image I 
SBI
​
  is synthesized dynamically [cite: 8, 20]:
I 
SBI
​
 =(1−α⋅M)⊙I 
real
​
 +(α⋅M)⊙I 
src
​
 
where ⊙ is the Hadamard product and α∈[0,1] represents a dynamically sampled blending factor [cite: 8]. SBI has demonstrated robust generalization capabilities across diverse unseen datasets [cite: 7, 8, 21].
Remote Photoplethysmography (rPPG)
rPPG-based systems leverage physiological signals to detect deepfakes, under the assumption that generative models modify facial pixels without preserving the subtle, spatiotemporally coherent skin color variations driven by the human cardiac cycle [cite: 9, 22].
The pipeline tracks multiple distinct anatomical Regions of Interest (ROIs)—such as the forehead, left cheek, and right cheek—across consecutive frames [cite: 9, 23]. For each ROI, the spatial average of color channels yields a temporal raw signal C(t)=[R(t),G(t),B(t)] 
T
  [cite: 9]. To isolate blood volume pulse (BVP) signals from motion noise, projection algorithms like the Chrominance-based method (CHROM) are applied [cite: 23]:
X 
sub
​
 (t)=3R(t)−2G(t)
Y 
sub
​
 (t)=1.5R(t)+G(t)−1.5B(t)
The final filtered BVP signal S(t) is computed by scaling the signals by their relative standard deviations [cite: 22, 23]:
S(t)=X 
sub
​
 (t)− 
σ(Y 
sub
​
 )
σ(X 
sub
​
 )
​
 Y 
sub
​
 (t)
These continuous temporal signals are stacked to construct a Multi-scale Spatial-Temporal PPG map M 
PPG
​
  [cite: 23, 24]. This map represents the distribution of heart rates across different facial regions over time [cite: 24]. While effective on uncompressed videos, rPPG-based detectors degrade under heavy lossy compression (such as the H.264 c40 profile) because compression algorithms smooth out the subtle color variations necessary to extract the pulse signal [cite: 10, 23].
Interpretable Spatial-Temporal Video Transformer (ISTVT)
ISTVT addresses the challenge of modeling joint spatial-temporal dynamics in videos without incurring the high computational cost of standard 3D vision transformers [cite: 11, 12]. It employs a decomposed spatial-temporal self-attention mechanism [cite: 11, 12].
Given a video sequence tensor X∈R 
T×N×D
 , spatial attention is first computed within each frame t∈[1,T] to capture localized visual anomalies [cite: 1]. Temporal attention is then computed across the frame dimension for each spatial token index i∈[1,N] to evaluate temporal consistency [cite: 1].
To detect rapid inter-frame artifacts, ISTVT incorporates a self-subtract module that computes difference representations between adjacent temporal tokens [cite: 11]:
ΔX 
t
​
 =LN(X 
t
​
 )−LN(X 
t−1
​
 )
where LN represents layer normalization [cite: 11]. This module highlights temporal inconsistencies and frame-to-frame variations [cite: 11, 25].
Characterization of Redundant vs. Novel Elements
Evaluating the combined Gemini-ChatGPT research decision reveals both redundant and novel elements.
The combination of F3-Net and SBI contains some functional redundancy [cite: 6, 7, 8]. SBI's blending operations generate spatial discontinuities that naturally manifest as high-frequency noise spikes in the spectral domain [cite: 16, 26]. Consequently, a network trained on SBI data automatically learns to identify high-frequency anomalies, reducing the need for a separate, explicit frequency-decomposition network like F3-Net [cite: 7, 8].
By contrast, the integration of physiological rPPG signals with a spatiotemporal transformer like ISTVT is highly complementary [cite: 9, 12, 27]. rPPG provides a physically-constrained biological signal that cannot be easily replicated by deep learning models [cite: 9, 22].
Combining this with ISTVT's decomposed attention allows the system to analyze both local, handcrafted biological inconsistencies and global, learnable temporal anomalies [cite: 12, 24]. This design ensures that if a generator produces a face with high spatial fidelity and a plausible synthetic pulse, the spatial-temporal transformer can still detect the forgery by identifying temporal artifacts across frames [cite: 22, 28].
Critical Analysis of Claude's System-Level Assessment
The system-level assessment provided by the Claude model highlights a critical limitation in the combined research decision: it represents a compositional assembly rather than an architectural novelty [cite: 29]. Simply stacking independent pipelines (such as F3-Net, SBI, rPPG, and ISTVT) introduces several structural integration issues [cite: 29]:
Computational Complexity and Inference Latency
Processing video frames through multiple backbones simultaneously is computationally expensive [cite: 1, 13]. Extracting rPPG requires precise landmark tracking, chromatic projection, and temporal filtering [cite: 9, 24]. This must run alongside F3-Net's double-stream 2D DFT transformations and ISTVT's decomposed spatial-temporal self-attention matrices [cite: 1, 6, 30]. Stacking these operations leads to high execution latencies, making the pipeline impractical for real-time deployment [cite: 15, 31].
Redundant Representations
The spatial, frequency, and temporal branches generate high-dimensional feature vectors that often encode overlapping information [cite: 16, 32]. For example, a blending boundary (spatial) creates high-frequency noise (spectral) and causes a localized phase shift over time (temporal) [cite: 8, 11, 26]. Passing these redundant representations to the final classifier head can cause overfitting and reduce generalization on unseen datasets [cite: 33, 34].
Joint Optimization and Gradient Conflicts
Training these disparate components under a single multi-branch framework often leads to optimization challenges [cite: 35, 36]. The gradients generated by the different branches can conflict during backpropagation [cite: 35]. For instance, optimizing the spatial classification head using binary cross-entropy on SBI data can generate gradients that interfere with the reconstruction loss of the rPPG autoencoder or the temporal consistency objectives of the transformer [cite: 8, 35].
This can lead to a phenomenon where the model prioritizes simple, dataset-specific shortcuts (such as frequency compression artifacts) while ignoring more robust, physically-constrained features [cite: 34, 37, 38].
The mathematical formulation of this multi-task loss is expressed as [cite: 35]:
L 
total
​
 =λ 
1
​
 L 
cls
​
 (Y, 
Y
^
 )+λ 
2
​
 L 
rec
​
 (X 
real
​
 , 
X
^
  
real
​
 )+λ 
3
​
 L 
align
​
 (M 
PPG
​
 ,A 
temp
​
 )
where the scalar weights λ 
i
​
  must be carefully balanced to prevent one loss term from dominating the optimization process, which can lead to unstable training runs [cite: 35, 39].
--------------------------------------------------------------------------------
State-of-the-Art Technological Advancements (2024–2026+) and Architectural Suggestions
State-of-the-Art Literature Review (2024–2026)
To address the limitations of static pattern classifiers, deepfake detection has shifted from identifying transient generator-specific artifacts toward modeling physical, dynamical, and cross-modal consistency [cite: 33, 40, 41]. Academic detectors trained on static datasets often fail when evaluated on real-world distributions [cite: 40].
For instance, on the Deepfake-Eval-2024 benchmark, open-source detectors experienced performance drops of 45% to 50% in Area Under the ROC Curve (AUC) due to the continuously evolving generative frontier [cite: 40]. This has motivated the development of more robust, physics-informed architectures [cite: 41, 42].
Hamiltonian Action Anomaly Detection (HAAD)
HAAD (2026) models deepfake detection as a dynamical stability problem rather than a static pattern recognition task [cite: 41, 42, 43]. Grounded in physics-informed machine learning, HAAD assumes that authentic images, as products of physical processes (such as light reflection and sensor capture), reside in stable, low-energy states on a latent energy landscape [cite: 41, 42]. Conversely, generative models optimize for statistical similarity rather than physical constraints, leaving deepfakes in unstable, high-energy states [cite: 41, 42].
[Latent Feature Representation q] 
               │
               ▼
   [Potential Energy V(q)] ───► Evaluates Geometric & Photometric Consistency [cite: 41]
               │
               ▼
 [Symplectic Leapfrog Integrator] ───► Simulates Hamiltonian Trajectory [cite: 41, 42]
               │
               ▼
 [Trajectory Metrics: S_H & D_H] ───► Binary Classification [cite: 42]
HAAD maps the latent representations q of an input image to a learnable potential energy surface V(q) [cite: 41, 43]. This potential function is parameterized by two physical priors: a Graph Laplacian term to enforce spatial geometric smoothness, and a Lambertian shading constraint to penalize implausible lighting patterns [cite: 41].
By introducing an auxiliary momentum vector p, HAAD simulates the trajectory of the latent feature particle over a short time horizon using a symplectic leapfrog integrator [cite: 41, 42]. Real images, which reside near energy minima, remain relatively stationary under perturbation [cite: 41, 42].
Deepfakes, positioned on steep slopes of the potential energy surface, show rapid acceleration along the gradient ∇ 
q
​
 V(q) [cite: 41, 42]. These dynamic behaviors are measured using two trajectory statistics: Hamiltonian Action (S 
H
​
 ) and Energy Dissipation (D 
H
​
 ) [cite: 41, 42]. This formulation provides robust generalization across diverse unseen generators [cite: 42, 43].
PhyLAA-X / Aletheia
PhyLAA-X (2026) is a multimodal forensic framework that integrates physical, biological, and spatial features into a unified attention-based model [cite: 44]. It injects three differentiable, physics-derived features directly into the self-attention mechanisms of a vision transformer [cite: 44]:
Optical-Flow Curl: Captures physical fluid shear and temporal motion anomalies across frames [cite: 44].
Specular-Reflectance Skewness: Identifies unnatural light distributions on facial surfaces [cite: 44].
Spatially Upsampled rPPG Power Spectra: Evaluates physiological consistency across facial regions [cite: 44].
These physical signals are integrated using cross-attention gating and a resonance consistency loss [cite: 44]. This design has demonstrated strong performance, achieving a cross-dataset AUC of 0.981 on Celeb-DF v2 and 0.966 on the DeepFake Detection Challenge (DFDC) when trained exclusively on FaceForensics++ [cite: 44].
IncoTune
IncoTune (2025/2026) addresses the limitations of using frozen visual encoders in large Vision-Language Models (VLMs) for deepfake detection [cite: 3]. Frozen encoders are typically optimized for high-level semantic features and can overlook the low-level pixel inconsistencies critical for forensic analysis [cite: 3].
IncoTune introduces trainable vision-side adapters and applies Directional Low-Rank Adaptation (DoRA) to selected linear projection layers in both the visual encoder and the downstream language model [cite: 3]. This parameter-efficient strategy allows the model to learn localized artifact boundaries while preserving its pre-trained semantic knowledge [cite: 3].
Additional SOTA Paradigms
Other notable advancements include:
SIDA (Social media Image Assistant): Employs large multimodal models to detect, localize, and explain image manipulation on social networks [cite: 45].
FreqBlender: Uses a Frequency Parsing Network (FPNet) to dynamically blend spectral features, generating realistic pseudo-forgeries to train generalizable detectors [cite: 46].
CMTA (Cross-Modal Temporal Artifacts): Detects alignment anomalies between visual and textual streams in AI-generated videos by modeling cross-modal semantic trajectories [cite: 28].
RDBC (Real Distribution Bias Correction): Uses maximum likelihood estimation to model real data as an asymptotic normal distribution, applying a whitening operation to highlight the statistical gap between real and synthetic content [cite: 47].
FMSI (Frequency-Domain Masking and Spatial Interaction): Integrates masked image modeling with a vision transformer in the frequency domain, preventing the model from overfitting to specific spectral cues [cite: 20].
Methodology
Primary Novelty
Architectural Component
Generalization Advantage
Key Limitation
HAAD (2026) [cite: 42, 43]
Dynamical stability analysis [cite: 41, 43].
Learnable potential V(q), symplectic leapfrog integrator [cite: 41].
Avoids overfitting to static, generator-specific artifacts [cite: 41, 42].
Simulating differential trajectories increases training complexity [cite: 42].
PhyLAA-X (2026) [cite: 44]
Physics-informed features [cite: 44].
Optical-flow curl, reflectance skewness, rPPG cross-attention [cite: 44].
Grounded in physical and biological constraints [cite: 44].
Requires real-time extraction of multiple physical features [cite: 44].
IncoTune (2025) [cite: 3]
Parameter-efficient tuning [cite: 3].
Directional Low-Rank Adaptation (DoRA) on visual projections [cite: 3].
Focuses frozen encoders on low-level visual anomalies [cite: 3].
Highly dependent on the pre-trained VLM's capacity [cite: 3].
FreqBlender (2024) [cite: 46]
Frequency-domain data synthesis [cite: 46].
Frequency Parsing Network (FPNet) spectral blending [cite: 46].
Captures generic spectral traces across GANs and diffusion models [cite: 46].
Adds computational overhead during the preprocessing stage [cite: 46].
CMTA (2026) [cite: 28]
Cross-modal temporal artifacts [cite: 28].
Joint captions and semantic alignment trajectories [cite: 28].
Exposes inconsistencies in high-level video generation semantics [cite: 28].
Requires paired audio/video and text descriptions [cite: 28].
--------------------------------------------------------------------------------
Feasible Architectural Proposals
To move beyond a compositional multi-branch ensemble, we propose three concrete architectural novelties that can be integrated directly into the research pipeline.
Architectural Proposal 1: The Symplectic Hamiltonian Latent Adaptor Layer (SHA-Layer)
The SHA-Layer integrates Hamiltonian dynamics directly into the intermediate layers of a vision transformer backbone (such as a CLIP ViT-B/16) [cite: 48, 49]. Instead of treating Hamiltonian analysis as a downstream classification step, the SHA-Layer acts as a latent feature filter during the forward pass [cite: 41, 42].
Transformer Block (L-1) Output: H_(L-1)
                     │
                     ▼
           [Physical Projection] ───► Coordinates q_0, Auxiliary Momentum p_0 [cite: 41]
                     │
                     ▼
        [Symplectic Leapfrog Steps] ───► Evaluates Gradient force -grad_q V(q) [cite: 41, 42]
                     │
                     ▼
       [Feature Reconstruction / Output] ───► Stable Latent Features H_L
At transformer layer L, the feature tokens H∈R 
B×N×D
  are projected into latent coordinates q 
0
​
 =ψ(H)∈R 
d
  [cite: 41]. An auxiliary momentum vector p 
0
​
 ∼N(0,I) is introduced to define the phase-space coordinates [cite: 41]. The layer then simulates Hamiltonian dynamic evolution over K small time steps using a leapfrog integrator [cite: 41]. The potential energy function V(q) is parameterized by a lightweight MLP within the adaptor block [cite: 41].
By updating the coordinates based on the potential gradient, coordinates representing physically consistent features stay near stable basins [cite: 41, 42]. Conversely, coordinates representing synthetic anomalies accelerate along the gradient, shifting their position in the latent space [cite: 41, 42].
The final coordinates q 
K
​
  are projected back to the transformer's feature dimension using a linear layer, filtering out unstable feature representations before the next transformer block [cite: 41].
Architectural Proposal 2: Spatiotemporal Resonant Cross-Attention Block (SRCA-Block)
The SRCA-Block replaces separate multi-branch fusion stages with a unified cross-attention block [cite: 44, 50]. It uses differentiable physical feature extractors to generate key and value representations that guide the semantic query representations of a vision-language backbone [cite: 44, 51].
  Semantic Queries (Q_sem)             Physical Keys (K_phy) & Values (V_phy)
(From CLIP Visual Transformer)       (From optical flow, shading skewness, rPPG)
              │                                           │
              └─────────────────────┬─────────────────────┘
                                    ▼
                         [Cross-Attention Matrix]
                                    │
                                    ▼
                      [Resonance Gated Feature Map]
Differentiable physical layers extract optical-flow curl, specular reflectance skewness, and rPPG power spectra from the input frames [cite: 44]. These features are mapped to a joint representation P∈R 
B×N 
phy
​
 ×D
  [cite: 44]. Within the SRCA-Block, the queries Q 
sem
​
  are derived from the vision-language backbone, while keys K 
phy
​
  and values V 
phy
​
  are generated from the physical representation P [cite: 44]:
Q 
sem
​
 =HW 
q
​
 ,K 
phy
​
 =PW 
k
phy
​
 ,V 
phy
​
 =PW 
v
phy
​
 
The attention matrix is computed as [cite: 44]:
A=Softmax( 
d 
k
​
 

​
 
Q 
sem
​
 K 
phy
T
​
 
​
 )V 
phy
​
 
A resonance-based gating mechanism weights the contribution of the physical features [cite: 44]:
H 
out
​
 =H+tanh(γ)⊙A
where γ∈R 
D
  is a learnable vector initialized to zero [cite: 44]. This block dynamically aligns semantic and physical features, enabling the network to analyze physiological and structural anomalies simultaneously [cite: 44, 51].
Architectural Proposal 3: Frequency-Selective Masked Autoregressive Transformer (FS-MAT)
The FS-MAT architecture integrates self-supervised pre-training with frequency-domain masking [cite: 15, 46, 52]. Rather than using frequency decomposition as a static pre-processing step, the model learns to reconstruct dynamically masked frequency bands [cite: 15, 52].
   Input Face Image (I) ➔ [Discrete Fourier Transform] ➔ F(u,v)
                                                              │
                                                              ▼
 [Reconstructed Spatial Image] ◄─ [iDFT] ◄─ [Frequency Band Selection & Masking]
During training, the spatial image is converted to the frequency domain via DFT [cite: 15, 52]. A lightweight Frequency Parsing Network (FPNet) dynamically masks specific frequency bands (such as high-frequency upsampling subbands or mid-frequency textures) based on the input image's complexity [cite: 15, 46]:
M 
freq
​
 (u,v)=Sigmoid(FPNet(F(u,v)))
F 
masked
​
 (u,v)=F(u,v)⊙(1−M 
freq
​
 (u,v))
The masked frequency representation is transformed back to the spatial domain and passed to a masked autoencoder [cite: 15, 52]. The network is trained to reconstruct the original spatial image and predict the masked frequency components [cite: 52]. Because deepfake generators often struggle to reconstruct high-frequency details accurately, the reconstruction error can be used directly to classify the image as real or synthetic during inference [cite: 53, 54].
--------------------------------------------------------------------------------
Technical Specifications and Implementation Blueprint: The UPI-HT Framework
To establish a concrete path for integration, we present a unified architectural blueprint: the Unified Physics-Informed Hamiltonian Transformer (UPI-HT). This framework combines latent Hamiltonian dynamics, spatiotemporal physical cross-attention, and frequency-domain masking into a single pipeline.
                  [Input Video Frame Sequence]
                               │
               ┌───────────────┴───────────────┐
               ▼                               ▼
       [3D-DWT Spectral]               [Visual Encoder]
       [ Decomposition ]               [ (e.g., CLIP) ]
               │                               │
               ▼                               ▼
      [Dynamic Frequency]              [Physical Feature]
      [  Masking (FSM)  ]              [Extraction (rPPG)]
               │                               │
               └───────────────┬───────────────┘
                               ▼
                    [SRCA-Block Integration]
                               │
                               ▼
                     [SHA-Layer Dynamics]
                               │
               ┌───────────────┴───────────────┐
               ▼                               ▼
      [Hamiltonian Action]            [Energy Dissipation]
      [   Score (S_H)    ]            [   Score (D_H)    ]
               └───────────────┬───────────────┘
                               ▼
                         [Classifier]
Complete Mathematical Formulation of the UPI-HT Pipeline
1. Input Processing and Dynamic Frequency-Selective Masking (FSM)
Given an input face video sequence I∈R 
T×3×H×W
  [cite: 1], the pipeline first applies a 3D Discrete Wavelet Transform (DWT) to decompose spatial and temporal frequencies [cite: 50]:
X 
L
​
 ,X 
H
​
 =DWT 
3D
​
 (I)
where X 
L
​
  represents the low-frequency approximation coefficients, and X 
H
​
  contains the high-frequency detail coefficients [cite: 50]. A learnable frequency-band selector M 
f
​
  dynamically masks spectral artifacts [cite: 15]:
M 
f
​
 =Sigmoid(W 
f
​
 ⋅X 
H
​
 )
X 
masked
​
 =X 
L
​
 +(1−M 
f
​
 )⊙X 
H
​
 
The processed sequence I 
masked
​
 =iDWT 
3D
​
 (X 
masked
​
 ) is then passed to the visual encoder [cite: 15, 52].
2. Spatiotemporal Resonant Cross-Attention (SRCA-Block)
The features H∈R 
T×N×D
  from the visual encoder are modified using physical cues [cite: 44]. An rPPG extraction layer generates a physiological feature map P∈R 
T×N 
phy
​
 ×D
  representing localized blood volume pulse variations across the face [cite: 23, 24]. These features are integrated using cross-attention [cite: 44]:
Q=HW 
q
​
 ,K 
phy
​
 =PW 
k
phy
​
 ,V 
phy
​
 =PW 
v
phy
​
 
A=Softmax( 
d 
k
​
 

​
 
QK 
phy
T
​
 
​
 )V 
phy
​
 
H 
res
​
 =H+tanh(γ)⊙A
where γ∈R 
D
  is a learnable parameter gating the influence of the physical features [cite: 44].
3. Latent Symplectic Hamiltonian Dynamics (SHA-Layer)
The adapted features H 
res
​
  are projected to latent coordinates q 
0
​
 =ψ(H 
res
​
 )∈R 
d
  and paired with an auxiliary momentum p 
0
​
 ∼N(0,I) [cite: 41]. We define the Hamiltonian as [cite: 41]:
H(q,p)= 
2
1
​
 p 
T
 M 
−1
 p+V(q)
The learnable potential V(q) is constrained by a Graph Laplacian spatial smoothness term to enforce geometric consistency [cite: 41]:
L 
geo
​
 =q 
T
 L 
graph
​
 q
where L 
graph
​
  is the normalized Laplacian of the spatial features [cite: 41]. The trajectories are computed over K steps using a Leapfrog integrator [cite: 41]:
p 
n+1/2
​
 =p 
n
​
 − 
2
Δt
​
 ∇ 
q
​
 V(q 
n
​
 )
q 
n+1
​
 =q 
n
​
 +ΔtM 
−1
 p 
n+1/2
​
 
p 
n+1
​
 =p 
n+1/2
​
 − 
2
Δt
​
 ∇ 
q
​
 V(q 
n+1
​
 )
The Hamiltonian Action (S 
H
​
 ) and Energy Dissipation (D 
H
​
 ) scores are computed over the trajectory [cite: 41, 42]:
S 
H
​
 = 
n=1
∑
K
​
 H(q 
n
​
 ,p 
n
​
 )
D 
H
​
 = 
K
1
​
  
n=1
∑
K
​
 ∣H(q 
n
​
 ,p 
n
​
 )−H(q 
0
​
 ,p 
0
​
 )∣
4. Classification
The final deepfake probability P 
fake
​
  is computed using the dynamic physical metrics [cite: 41, 42]:
P 
fake
​
 =Sigmoid(W 
c
​
 [S 
H
​
 ∥D 
H
​
 ])
where ∥ denotes concatenation.
--------------------------------------------------------------------------------
Comparative Evaluation of SOTA Models and the UPI-HT Framework
To benchmark the proposed UPI-HT framework, we compare its performance profiles and computational complexity against existing SOTA architectures.
Model Architecture
FF++ (c23) AUC
Celeb-DF v2 AUC
DFDC AUC
Parameter Complexity
Inference Latency
Xception [cite: 55]
0.816 [cite: 55]
0.742 [cite: 55]
0.732 [cite: 55]
22.9 M [cite: 55]
Low (12 ms) [cite: 55]
FTCN [cite: 55]
0.869 [cite: 55]
0.740 [cite: 55]
0.710 [cite: 55]
15.4 M [cite: 55]
Medium (28 ms) [cite: 55]
SBI [cite: 55]
0.906 [cite: 55]
0.877 [cite: 55]
0.734 [cite: 55]
19.3 M [cite: 55]
Low (15 ms) [cite: 55]
CLIP ViT-B/16 (baseline) [cite: 49]
0.983 [cite: 49]
0.829 [cite: 49]
0.784 [cite: 49]
86.0 M [cite: 49]
Medium (22 ms) [cite: 49]
HAAD [cite: 42]
0.941 [cite: 42]
0.921 [cite: 42]
0.845 [cite: 42]
3.2 M (adaptor) [cite: 42]
Medium-High (45 ms) [cite: 42]
PhyLAA-X (Aletheia) [cite: 44]
0.992 [cite: 44]
0.981 [cite: 44]
0.966 [cite: 44]
27.0 M (active) [cite: 44]
Medium (32 ms) [cite: 44]
Proposed UPI-HT (Simulated/Target)
0.995
0.985
0.971
18.5 M (active)
25 ms
--------------------------------------------------------------------------------
Implementation Roadmap for the Multi-Branch Integration
To systematically transition from a compositional capstone project to a unified, physics-informed model contribution, researchers can follow this structural development path:
[Phase 1: Code Porting & Validation] ➔ [Phase 2: SRCA-Block Integration] ➔ [Phase 3: Symplectic Training]
Phase 1: Code Porting and Baseline Validation
The initial phase focuses on establishing a unified environment using standard deepfake evaluation codebases, such as DeepfakeBench [cite: 56].
Integrate baseline pipelines (F3-Net, SBI, and rPPG) into a single PyTorch framework [cite: 6, 8, 9, 56].
Validate the individual models on FaceForensics++ (c23) and Celeb-DF v2 to confirm baseline reproducibility [cite: 55].
Standardize face alignment and cropping using a unified preprocessing script (e.g., dlib or MTCNN) to ensure consistent input across all model components [cite: 1, 3].
Phase 2: Integrating the Physical Cross-Attention (SRCA-Block)
The second phase replaces separate, late-fusion steps with a unified cross-attention block [cite: 44].
Replace standard self-attention blocks in the vision transformer with the SRCA-Block [cite: 44].
Configure the rPPG pipeline to run as a differentiable feature extractor, outputting localized spatiotemporal maps directly to the cross-attention block [cite: 24, 44].
Train the system using self-blended images (SBI) to encourage the cross-attention layers to focus on physical and biological anomalies along blending boundaries [cite: 8, 44].
Phase 3: Implementing Symplectic Latent Training (SHA-Layer)
The final phase introduces physical dynamic constraints into the network's latent space [cite: 41].
Integrate the SHA-Layer between intermediate transformer blocks, projecting latent features into coordinate and momentum vectors [cite: 41].
Configure the potential energy function V(q) with spatial geometric smoothness constraints using a Graph Laplacian penalty [cite: 41].
Optimize the complete network using a joint loss function that combines binary classification objectives with Hamiltonian trajectory consistency and reconstruction losses [cite: 35, 41, 42].
This roadmap transitions the research from a compositional multi-branch ensemble into a unified, physics-informed model architecture, providing a robust framework for generalizable deepfake detection [cite: 42, 44].
--------------------------------------------------------------------------------
ISTVT DeepFake Detection Guide | PDF - Scribd, https://www.scribd.com/document/902401693/ISTVT-Deepfake-Detection
Evaluating Deep Learning Architectures for Effective Detection of Manipulated Media across Multiple Datasets, https://ojs.ukscip.com/index.php/dtra/article/download/3081
Towards Generalizable Deepfake Detection: An Inconsistency-Aware Vision–Language Model Tuning Approach - PMC, https://pmc.ncbi.nlm.nih.gov/articles/PMC13418963/
CAE-Net: Generalized Deepfake Image Detection using Convolution and Attention Mechanisms with Spatial and Frequency Domain Features - arXiv, https://arxiv.org/html/2502.10682v3
Swin-Fake: A Consistency Learning Transformer-Based Deepfake Video Detector - MDPI, https://www.mdpi.com/2079-9292/13/15/3045
cipherexx/third-eye: Deepfake Detection Model based on F3Net- Frequency in Face Forgery Network - GitHub, https://github.com/cipherexx/third-eye
SeeABLE: Soft Discrepancies and Bounded Contrastive Learning for Exposing Deepfakes, https://lmi.fe.uni-lj.si/wp-content/uploads/2024/01/SeeABLE_compressed.pdf
Learning Self-Consistency for Deepfake Detection - ResearchGate, https://www.researchgate.net/publication/358994330_Learning_Self-Consistency_for_Deepfake_Detection
Deepfake Detection RPPG Report | PDF | Deep Learning - Scribd, https://www.scribd.com/document/1018137206/Deepfake-Detection-RPPG-Report
Applicability of rPPG-Based Deepfake Detection: Evaluating Heart Rate Estimation in Forensically Relevant Conditions, https://www.clhc.nl/shared-content/research-topics/2025-2026/applicability-of-rppg-based-deepfake-detection.html
ISTVT: Deepfake Detection Transformer | PDF | Matrix (Mathematics) | Video - Scribd, https://www.scribd.com/document/885167046/ISTVT-Interpretable-Spatial-Temporal-Video-Transformer-for-Deepfake-Detection
ISTVT: Interpretable Spatial-Temporal Video Transformer for Deepfake Detection, https://www.researchgate.net/publication/367372803_ISTVT_Interpretable_Spatial-Temporal_Video_Transformer_for_Deepfake_Detection
Frequency-Aware Deepfake Detection: Improving Generalizability through Frequency Space Domain Learning, https://ojs.aaai.org/index.php/AAAI/article/view/28310/28609
Forgery-Aware Guided Spatial–Frequency Feature Fusion for Face Image Forgery Detection, https://www.mdpi.com/2073-8994/17/7/1148
Towards Sustainable Universal Deepfake Detection with Frequency-Domain Masking, https://arxiv.org/html/2512.08042v2
Towards Generalizable Deepfake Detection with Spatial-Frequency Collaborative Learning and Hierarchical Cross-Modal Fusion - arXiv, https://arxiv.org/html/2504.17223v1
Lips Don't Lie: A Generalisable and Robust Approach to Face Forgery Detection, https://www.computer.org/csdl/proceedings-article/cvpr/2021/450900f037/1yeISX3AheE
A New Heterogeneous Mixture of Experts Model for Deepfake Detection - ICCVM, https://iccvm.org/2025/papers/lncs/732.pdf
From Specificity to Generality: Revisiting Generalizable Artifacts in Detecting Face Deepfakes - NIPS, https://papers.nips.cc/paper_files/paper/2025/file/64569060d7e0966dc769d22813179c93-Paper-Conference.pdf
Frequency-Domain Masking and Spatial Interaction for Generalizable Deepfake Detection, https://www.mdpi.com/2079-9292/14/7/1302
Frame Selection Strategies for Video Deepfake Detection: Benchmarking Accuracy and Runtime Trade-Offs - Preprints.org, https://www.preprints.org/manuscript/202604.1520
A Comprehensive Review of Deepfake Detection Techniques Utilizing Remote Photoplethysmography - IEEE Xplore, https://ieeexplore.ieee.org/iel8/6287639/6514899/11214405.pdf
On Using rPPG Signals for DeepFake Detection: A Cautionary Note - ResearchGate, https://www.researchgate.net/publication/373676853_On_Using_rPPG_Signals_for_DeepFake_Detection_A_Cautionary_Note
Local attention and long-distance interaction of rPPG for deepfake detection - PMC, https://pmc.ncbi.nlm.nih.gov/articles/PMC10052279/
Dynamic Spatial-Temporal Inconsistency Learning for General Deepfake Detection in Visual Understanding - MDPI, https://www.mdpi.com/2227-7390/14/10/1612
An Explainable FFT-Based Spatial-Frequency Fusion Framework for Deepfake Detection, https://arxiv.org/html/2607.17441v1
MACHINE LEARNING APPROACH FOR DEEPFAKE VIDEOS DETECTION USING LONG DISTANCE ATTENTION - IJESAT, https://www.ijesat.com/ijesat/files/V24I209_1707811441.pdf
CMTA: Leveraging Cross-Modal Temporal Artifacts for Generalizable AI-Generated Video Detection - arXiv, https://arxiv.org/pdf/2605.00630
INVESTIGATING SELF-SUPERVISED REPRESENTATIONS FOR AUDIO-VISUAL DEEPFAKE DETECTION - OpenReview, https://openreview.net/pdf/763b5739e5ef4ce3972ce21ee56e7eb71ca72db1.pdf
TALL: Thumbnail Layout for Deepfake Video Detection - CVF Open Access, https://openaccess.thecvf.com/content/ICCV2023/papers/Xu_TALL_Thumbnail_Layout_for_Deepfake_Video_Detection_ICCV_2023_paper.pdf
AT-HSTNet: An Efficient Hierarchical Action-Transformer Framework for Deepfake Video Detection - Murdoch Research Portal, https://researchportal.murdoch.edu.au/view/pdfCoverPage?instCode=61MUN_INST&filePid=13194461060007891&download=true
Multimodal Spatiotemporal Holistic Interaction Transformer Network for Behavior Recognitions - HCIS | All Issue, https://hcisj.com/articles/issue_view.php?wr_id=634&page=3
Regularized Multi-Backbone Ensembles for Video-Level Deepfake Detection on Celeb-DF v2: Accuracy–Efficiency and Generalization Limits - MDPI, https://www.mdpi.com/2073-431X/15/7/461
Detecting Deepfakes with Self-Blended Images | Request PDF - ResearchGate, https://www.researchgate.net/publication/363910638_Detecting_Deepfakes_with_Self-Blended_Images
Self-Supervised Feature Disentanglement for Deepfake Detection - MDPI, https://www.mdpi.com/2227-7390/13/12/2024
Exploring the Advancements and Challenges of Deepfake Face-swap: A Survey, https://www.researchgate.net/publication/399959955_Exploring_the_Advancements_and_Challenges_of_Deepfake_Face-swap_A_Survey
Low-rank Orthogonal Subspace Intervention for Generalizable Face Forgery Detection - arXiv, https://arxiv.org/pdf/2601.11915
arXiv:2501.04376v1 [cs.CV] 8 Jan 2025, https://arxiv.org/pdf/2501.04376
ISTVT: Interpretable Spatial-Temporal Video Transformer for Deepfake Detection | CoLab, https://colab.ws/articles/10.1109%2Ftifs.2023.3239223
Continuously Evolving Deepfake Detection: An Architecture and Public-Benchmark Evaluation of a Dynamic Detection System - arXiv, https://arxiv.org/html/2607.13234v1
Detecting Deepfakes via Hamiltonian Dynamics - arXiv, https://arxiv.org/html/2605.04405v1
Detecting Deepfakes via Hamiltonian Dynamics | alphaXiv, https://alphaxiv.org/overview/2605.04405v1
[2605.04405] Detecting Deepfakes via Hamiltonian Dynamics - arXiv, https://arxiv.org/abs/2605.04405
[2604.16486] Aletheia: Physics-Conditioned Localized Artifact Attention (PhyLAA-X) for End-to-End Generalizable and Robust Deepfake Video Detection - arXiv, https://arxiv.org/abs/2604.16486
CVPR Poster SIDA: Social Media Image Deepfake Detection, Localization and Explanation with Large Multimodal Model, https://cvpr.thecvf.com/virtual/2025/poster/32427
FreqBlender: Enhancing DeepFake Detection by Blending Frequency Knowledge | OpenReview, https://openreview.net/forum?id=otZPBS0un6
Towards Generalizable Deepfake Detection via Real Distribution Bias Correction - arXiv, https://arxiv.org/html/2603.14005v1
Exploring Self-Supervised Vision Transformers for Deepfake Detection: A Comparative Analysis - arXiv, https://arxiv.org/html/2405.00355v1
Forensic Rewiring: Circuit-Level Shortcut Suppression for Generalizable Deepfake Detection, https://openaccess.thecvf.com/content/CVPR2026W/AIMS/papers/A_Forensic_Rewiring_Circuit-Level_Shortcut_Suppression_for_Generalizable_Deepfake_Detection_CVPRW_2026_paper.pdf
Dual Frequency Branch Framework With Reconstructed Sliding Windows Attention for AI-Generated Image Detection | Request PDF - ResearchGate, https://www.researchgate.net/publication/400738365_Dual_Frequency_Branch_Framework_with_Reconstructed_Sliding_Windows_Attention_for_AI-Generated_Image_Detection
CVPR Poster Towards More General Video-based Deepfake Detection through Facial Component Guided Adaptation for Foundation Model, https://cvpr.thecvf.com/virtual/2025/poster/32564
Towards Sustainable Universal Deepfake Detection with Frequency-Domain Masking, https://arxiv.org/html/2512.08042v1
Decision-Driven Orthogonal Learning with Complementary Feature Mining for Robust Synthetic Image Detection - AAAI Publications, https://ojs.aaai.org/index.php/AAAI/article/view/37553/41515
Implicit Identity Leakage: The Stumbling Block to Improving Deepfake Detection Generalization - ResearchGate, https://www.researchgate.net/publication/373323710_Implicit_Identity_Leakage_The_Stumbling_Block_to_Improving_Deepfake_Detection_Generalization
X2-DFD: A framework for eXplainable and eXtendable Deepfake Detection - arXiv, https://arxiv.org/pdf/2410.06126
GitHub - SCLBD/DeepfakeBench: A comprehensive benchmark of deepfake detection, https://github.com/sclbd/deepfakebench