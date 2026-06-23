# vllm GGUF qwen35/MTP Support - TODO

## Kontext
Das Ziel ist, den vllm-Fork (`/spinning/llm_stuff/vllm_gguf_fork/`) so zu erweitern, dass GGUF-Dateien für Huihui-Qwen3.6-27B-abliterated-MTP funktionieren:

```
Model GGUF: /spinning/llm_stuff/club-3090/models-cache/huihui-ai/Huihui-Qwen3.6-27B-abliterated-MTP-GGUF/Huihui-Qwen3.6-27B-abliterated-ggml-model-Q8_0.gguf
MMProj GGUF: /spinning/llm_stuff/club-3090/models-cache/huihui-ai/Huihui-Qwen3.6-27B-abliterated-MTP-GGUF/mmproj-model-f16.gguf
```

### GGUF-Datei-Analyse
- **Architektur**: `qwen35` (GGUF naming)
- **Layers**: 65 (0-63 Main + 64 MTP)
- **SSM Tensors** (pro Layer, idx 0-63):
  - `blk.N.ssm_a`
  - `blk.N.ssm_alpha.weight`
  - `blk.N.ssm_beta.weight`
  - `blk.N.ssm_conv1d.weight`
  - `blk.N.ssm_dt.bias`
  - `blk.N.ssm_norm.weight`
  - `blk.N.ssm_out.weight`
  - `blk.N.attn_gate.weight`
- **MTP Tensors** (Layer 64):
  - `blk.64.nextn.shared_head_norm.weight`
  - `blk.64.nextn.enorm.weight`
  - `blk.64.nextn.hnorm.weight`
  - `blk.64.nextn.eh_proj.weight`
- **GGUF Metadata**:
  - `qwen35.nextn_predict_layers = 1`
  - `qwen35.ssm.conv_kernel = [4]`
  - `qwen35.ssm.state_size = [128]`
  - `qwen35.ssm.group_count = [16]`
  - `qwen35.ssm.time_step_rank = [48]`
  - `qwen35.ssm.inner_size = [6144]`

---

## ✅ Phase 1: Analyse & Vorbereitung (ABGESCHLOSSEN)

- [x] GGUF-Datei inspiziert (Magic, Version, Tensor-Count)
- [x] Tensor-Namen analysiert (SSM, MTP, Standard-Layer)
- [x] GGUF-Metadaten extrahiert
- [x] vllm vom upstream geforkt nach `vllm_gguf_fork/`
- [x] Bestehende qwen3_5 Implementierung im KVarN-Fork analysiert
- [x] Model-Registry Einträge für qwen3_5 vorhanden

---

## ✅ Phase 2: GGUF <→ qwen35 Mapping (ABGESCHLOSSEN)

### 2.1 Tensor-Namen Mapping (`gguf_loader.py`)

- [x] **qwen35 model_type Handling**:
  - Mapping `qwen3_5_text` → `qwen35` (GGUF naming)
  - Mapping `qwen3_5_moe_text` → `qwen35moe`

- [x] **SSM Tensor Mapping** (pro Layer):
  - `blk.N.ssm_a` → `model.layers.N.ssm_a`
  - `blk.N.ssm_alpha.weight` → `model.layers.N.ssm_alpha.weight`
  - `blk.N.ssm_beta.weight` → `model.layers.N.ssm_beta.weight`
  - `blk.N.ssm_conv1d.weight` → `model.layers.N.ssm_conv1d.weight`
  - `blk.N.ssm_dt.bias` → `model.layers.N.ssm_dt.bias`
  - `blk.N.ssm_norm.weight` → `model.layers.N.ssm_norm.weight`
  - `blk.N.ssm_out.weight` → `model.layers.N.ssm_out.weight`
  - `blk.N.attn_gate.weight` → `model.layers.N.attn_gate.weight`

- [x] **MTP Tensor Mapping**:
  - `blk.64.nextn.shared_head_norm.weight` → `model.layers.64.nextn.shared_head_norm.weight`
  - `blk.64.nextn.enorm.weight` → `model.layers.64.nextn.enorm.weight`
  - `blk.64.nextn.hnorm.weight` → `model.layers.64.nextn.hnorm.weight`
  - `blk.64.nextn.eh_proj.weight` → `model.layers.64.nextn.eh_proj.weight`

- [x] **MTP Layer in sideload_params skippen**

### 2.2 GGUF Metadata Patching (`gguf_utils.py`)

- [x] **MTP Config Patching**:
  - `qwen35.nextn_predict_layers` → `mtp_num_hidden_layers`

- [x] **SSM Config Patching**:
  - `qwen35.ssm.conv_kernel` → `ssm_conv_kernel`
  - `qwen35.ssm.state_size` → `ssm_state_size`
  - `qwen35.ssm.group_count` → `ssm_group_count`
  - `qwen35.ssm.time_step_rank` → `ssm_time_step_rank`
  - `qwen35.ssm.inner_size` → `ssm_inner_size`

### 2.3 MoE Handling (für qwen35 MoE Varianten)

- [x] `qwen2_moe`, `qwen3_moe`, `qwen3_5_moe` Expert Weights Mapping

---

## ✅ Phase 3: Model-Registrierung (ABGESCHLOSSEN)

- [x] `Qwen3_5ForConditionalGeneration` → `(qwen3_5, Qwen3_5ForConditionalGeneration)`
- [x] `Qwen3_5MoeForConditionalGeneration` → `(qwen3_5, Qwen3_5MoeForConditionalGeneration)`
- [x] `Qwen3_5MTP` → `(qwen3_5_mtp, Qwen3_5MTP)`
- [x] `Qwen3_5MoeMTP` → `(qwen3_5_mtp, Qwen3_5MoeMTP)`

---

## ⏳ Phase 4: Docker Build & Deployment (IN PROGRESS)

- [ ] **Docker Build abwarten** (läuft seit ~40min, C++ Kompilierung)
- [ ] Container: `vllm-gguf:latest` in `/spinning/llm_stuff/vllm_gguf_fork/`

---

## 🔲 Phase 5: Hardware-Test auf RTX 5090

### 5.1 Grundlegender GGUF-Load Test
- [ ] vllm mit GGUF-Datei starten:
  ```bash
  docker run --gpus '"device=0"' -v /spinning/llm_stuff/club-3090/models-cache:/models \
    vllm-gguf:latest --model /models/huihui-ai/Huihui-Qwen3.6-27B-abliterated-MTP-GGUF/ \
    --gguf-model-file Huihui-Qwen3.6-27B-abliterated-ggml-model-Q8_0.gguf \
    --trust-remote-code
  ```
- [ ] Erwartet: Model lädt ohne Tensor-Mismatch Fehler
- [ ] Test-Prompt senden → prüfen ob Output generiert wird

### 5.2 MTP-Funktionalität Test
- [ ] Prüfen ob MTP-Layer erkannt wird (Log-Level INFO)
- [ ] SSM-Parameter werden korrekt geladen
- [ ] nextn_predict_layers = 1 wird gelesen

### 5.3 Multimodal-Test (mit mmproj)
- [ ] mmproj GGUF wird erkannt und geladen
- [ ] Vision-Encoding funktioniert (falls Bild-Input)

### 5.4 Performance Benchmarks
- [ ] throughput tokens/sec messen
- [ ] Memory Usage (VRAM)
- [ ] Vergleich mit PyTorch-Benchmark (falls vorhanden)

---

## 🔲 Phase 6: Fehlerbehebung & Iterationen

### Bekannte potentielle Probleme:

1. **SSM Layer werden nicht erkannt**
   - Symptom: "Tensor not found" Fehler
   - Lösung: Mapping in gguf_loader.py prüfen

2. **MTP Layer 64 verursacht Dimension-Mismatch**
   - Symptom: Shape mismatch beim Laden
   - Lösung: num_hidden_layers anpassen (erwartet 64, nicht 32)

3. **rope.scaling Config fehlt**
   - Symptom: RoPE Config nicht gefunden
   - Lösung: `llama.rope.scaling` → `qwen35.rope.scaling` Mapping

4. **Quantisierung Q8_0 nicht unterstützt**
   - Symptom: quantization Fehler
   - Lösung: GGUF quantization support prüfen

5. **attn_qkv.weight Naming (GGUF vs HF)**
   - GGUF: `blk.N.attn_qkv.weight` (fused QKV)
   - HF: `blk.N.attn_q.weight`, `blk.N.attn_k.weight`, `blk.N.attn_v.weight`
   - Lösung: Unfusing in weight loading

---

## 🔲 Phase 7: Upstream Contribution (OPTIONAL)

- [ ] PR an upstream vllm-project/vllm mit qwen35 GGUF Support
- [ ] Tests hinzufügen
- [ ] Dokumentation aktualisieren

---

## Änderungsübersicht

### Geänderte Dateien:
1. `vllm/model_executor/model_loader/gguf_loader.py` - Tensor Mappings
2. `vllm/transformers_utils/gguf_utils.py` - Config Patching
3. `vllm/model_executor/models/qwen3_5.py` - Model-Registrierung
4. `vllm/model_executor/models/qwen3_5_mtp.py` - MTP Model
5. `vllm/transformers_utils/configs/qwen3_5.py` - Config
6. `vllm/transformers_utils/configs/qwen3_5_moe.py` - MoE Config

### Neue Dateien im Fork:
- Dockerfile + docker-compose.yml

---

## Checkliste für Deployment

```bash
# 1. Build prüfen
docker images | grep vllm-gguf

# 2. Container starten
docker compose -f /spinning/llm_stuff/vllm_gguf_fork/docker-compose.yml up -d

# 3. Model laden
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen3.5-27b-mtp", "prompt": "Hello", "max_tokens": 50}'

# 4. Logs prüfen
docker compose -f /spinning/llm_stuff/vllm_gguf_fork/docker-compose.yml logs -f
```
