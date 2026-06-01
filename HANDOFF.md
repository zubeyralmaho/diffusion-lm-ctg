# Berkay için — Colab Eğitim Handoff'u

Selam Berkay. Bu repo CENG 467 dönem projesi (Controllable Text Generation via
Diffusion Models). Tüm kod hazır, sadece **Colab Pro'da eğitimleri çalıştırıp
sonuçları geri göndermen** lazım.

## Yapacakların — özet

1. Colab Pro'da `notebooks/colab_train.ipynb` aç (T4/A100/L4 fark etmez, A100 varsa süper)
2. Hücreleri sırayla çalıştır
3. En son hücre `results.zip` üretir — onu indir, Zübeyr'e (WhatsApp / Drive) gönder

## Adım adım

### 1. Notebook'u aç

[colab.research.google.com](https://colab.research.google.com) → File → Open notebook → GitHub sekmesi → `zubeyralmaho/diffusion-lm-ctg` yaz → `notebooks/colab_train.ipynb` seç.

### 2. GPU seç

Runtime → Change runtime type → **A100** (varsa) veya **L4**, yoksa T4.

### 3. Hücreleri çalıştır

Sırayla, üstten aşağı. Her hücre öncekinin bitmesini bekler.

Yeni notebook akışında iki önemli güvenlik katmanı var:

- Her major stage sonunda artifact snapshot'ı otomatik olarak Google Drive altına kaydedilir.
- Her komut canlı log verir; ayrıca exit code ve süre bilgisi Drive'da status JSON olarak tutulur.

Bu yüzden **0b. Drive mount + run klasörü** hücresini atlama.

Snapshot dizini şu formatta oluşur:

```text
MyDrive/diffusion-lm-ctg-runs/<RUN_ID>/
```

Altında üç önemli klasör bulunur:

- `status/` — her stage için süre ve exit code
- `artifacts/` — checkpoint, generation, metric ve zip snapshot'ları
- `logs/` — canlı komut çıktısının kaydı

Eğer Colab runtime'ı düşerse yeni oturumda sadece şu sırayı izle:

1. 0b, 1 ve 1b hücrelerini tekrar çalıştır
2. Notebook'un en altındaki restore hücresinde `RESTORE_RUN_ID` ve `RESTORE_STAGE` alanlarını doldur
3. Kaldığın yerden devam et

**Önemli:** Bölüm 1b'deki **smoke test** hücresi (~10 sn) ilk önce çalışsın. 4 testin de PASS olmalı. Bu pipeline'ın bozulmadığını doğrular — saatler harcamadan önce. FAIL alırsan Zübeyr'e haber ver.

Tahmini süreler:

| Hücre | T4 | A100 |
|---|---|---|
| Setup + clone + pip install | 2 dk | 2 dk |
| `prepare_data.sh` | 1 dk | 1 dk |
| GPT-2 baseline train (5 epoch) | 25 dk | 8 dk |
| T5 baseline train (5 epoch) | 30 dk | 10 dk |
| Diffusion-LM train (20 epoch) | 2-3 saat | 45 dk |
| `run_eval.sh` (generation + metrikler) | 30 dk | 12 dk |
| `run_ablations.sh` (5 ablation) | 3-4 saat | 1.5 saat |
| **Toplam** | **~7 saat** | **~3 saat** |

### 4. Sonuç paketleme

En son hücre `results.zip` indirir (içinde metrikler, generation örnekleri,
özet tablo). Bu zip'i Zübeyr'e gönder.

Eğer checkpoint'leri de Drive'a almak istersen Drive mount hücresi var.

## Sorun çıkarsa

- **OOM (out of memory)**: ilgili config'de `batch_size`'ı yarıya düşür (örn. 64 → 32).
- **Runtime disconnect**: Colab Pro'da nadir, ama olursa kaldığın hücreden devam et — checkpoint'ler her epoch sonunda kaydediliyor.
- **`No module named X`**: ilk pip install hücresini tekrar çalıştır.
- **Başka bir şey**: Zübeyr'e yaz, repo yapısını birlikte gözden geçiririz.

## Eğer zamanın kısıtlıysa — minimum koşu

Tüm pipeline çok uzun gelirse **sadece şunları çalıştır** (~1.5 saat T4'te):

1. Setup hücreleri
2. `prepare_data.sh`
3. GPT-2 baseline train + eval
4. T5 baseline train + eval
5. Son zip hücresi

Diffusion-LM ve ablation'ları Zübeyr ikinci oturumda yapar. Bu bile progress
report için yeterli rakam verir.
