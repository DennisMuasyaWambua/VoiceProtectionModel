"""Speaker encoders. Seen (training/selection): SpeechBrain ECAPA-TDNN (VoxCeleb).
Held-out (evaluation only): WavLM-Base+ x-vector verifier (microsoft/wavlm-base-plus-sv) and Resemblyzer GE2E."""
import numpy as np, torch, torch.nn as nn

class ECAPA(nn.Module):
    """Differentiable wrapper around speechbrain/spkrec-ecapa-voxceleb."""
    def __init__(self, savedir="pretrained/ecapa"):
        super().__init__()
        from speechbrain.inference.speaker import EncoderClassifier
        self.clf = EncoderClassifier.from_hparams("speechbrain/spkrec-ecapa-voxceleb", savedir=savedir, run_opts={"device": "cpu"})
        self.clf.eval()
        for p in self.clf.mods.parameters(): p.requires_grad_(False)
    def forward(self, wav):  # (B,N) → (B,192)
        lens = torch.ones(wav.shape[0])
        f = self.clf.mods.compute_features(wav)
        f = self.clf.mods.mean_var_norm(f, lens)
        return self.clf.mods.embedding_model(f, lens).squeeze(1)

class WavLMSV:
    def __init__(self):
        from transformers import AutoFeatureExtractor, WavLMForXVector
        self.fe = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")
        self.m = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv").eval()
    @torch.no_grad()
    def __call__(self, wavs):
        out = []
        for w in wavs:
            w = w.detach().cpu().numpy() if torch.is_tensor(w) else w
            inp = self.fe(w, sampling_rate=16000, return_tensors="pt")
            out.append(self.m(**inp).embeddings[0])
        return torch.nn.functional.normalize(torch.stack(out), dim=-1)

class Resemblyzer:
    def __init__(self):
        from resemblyzer import VoiceEncoder
        self.enc = VoiceEncoder("cpu", verbose=False)
    def __call__(self, wavs):
        from resemblyzer import preprocess_wav
        out = []
        for w in wavs:
            w = w.detach().cpu().numpy() if torch.is_tensor(w) else w
            out.append(torch.from_numpy(self.enc.embed_utterance(preprocess_wav(w.astype(np.float32), 16000))))
        return torch.stack(out)
