# Colab Pro+ Runbook — Kendi Notlarım

CENG 467 dönem projesi (Controllable Text Generation via Diffusion Models).
Tüm kod hazır; bu döküman Colab Pro+ üzerinde eğitimi koşarken kendime
hatırlatıcı.

## Akış — özet

1. Colab Pro+'ta `notebooks/colab_train.ipynb` aç (A100 tercih)
2. Hücreleri sırayla çalıştır
3. En son hücre `results.zip` üretir — indir, repo'daki `results/`'a koy

## Adım adım

### 1. Notebook'u aç

[colab.research.google.com](https://colab.research.google.com) → File → Open
notebook → GitHub sekmesi → `zubeyralmaho/diffusion-lm-ctg` →
`notebooks/colab_train.ipynb`.

Setup hücresi repo zaten varsa `git pull --ff-only` ile en güncel commit'i
çeker.

### 2. GPU seç

Runtime → Change runtime type → **A100** (Pro+ ile öncelikli erişim var),
yoksa L4. Background execution Pro+'ta açık; uzun ablation koşusu için
bunu aktif et.

### 3. Hücreleri çalıştır

Sırayla, üstten aşağı. İki güvenlik katmanı:

- Her major stage sonunda artifact snapshot'ı Google Drive altına yazılır.
- Her komut canlı log verir; exit code ve süre Drive'da status JSON olarak
  tutulur.

Bu yüzden **0b. Drive mount + run klasörü** hücresini atlama.

Snapshot dizini:

```text
MyDrive/diffusion-lm-ctg-runs/<RUN_ID>/
```

İçinde:

- `status/` — her stage için süre ve exit code
- `artifacts/` — checkpoint, generation, metric ve zip snapshot'ları
- `logs/` — canlı komut çıktısı

Runtime düşerse yeni oturumda:

1. 0b, 1 ve 1b hücrelerini tekrar çalıştır
2. Notebook'un en altındaki restore hücresinde `RESTORE_RUN_ID` ve
   `RESTORE_STAGE` alanlarını doldur
3. Kaldığın yerden devam et

Diffusion-LM için ek not:

- Son düzeltmeler: sampler, target-mask, pretrained embedding init,
  self-conditioning, EMA ve classifier guidance.
- Eski diffusion checkpoint'leri **geçersiz** — atılmalı.
- Notebook'taki diffusion hücresi eski `checkpoints/diffusion_lm` klasörünü
  ve diffusion output'larını temizleyip sıfırdan koşar.

**Önemli:** Bölüm 1b'deki **smoke test** hücresi (~10 sn) önce çalışsın.
4 testin de PASS olmalı. FAIL alırsam saatler harcamadan önce sebebini
çözmem lazım.

Tahmini süreler (A100 / Pro+):

| Stage | A100 |
|---|---|
| Setup + clone + pip install | 2 dk |
| `prepare_data.sh` | 1 dk |
| GPT-2 baseline train (5 epoch) | 8 dk |
| T5 baseline train (5 epoch) | 10 dk |
| Diffusion-LM train (20 epoch) | 45 dk |
| `run_eval.sh` (generation + metrikler) | 12 dk |
| `run_ablations.sh` (5 ablation) | 1.5 saat |
| **Toplam** | **~3 saat** |

### 4. Sonuç paketleme

En son hücre `results.zip` üretir (metrikler, generation örnekleri, özet
tablo). İndir, repo köküne `results/` olarak aç ve commit'le.

Checkpoint'ler Drive'da snapshot olarak duruyor — gerekirse oradan çekilir.

## Sorun çıkarsa

- **OOM**: ilgili config'de `batch_size`'ı yarıya düşür (örn. 64 → 32).
- **Runtime disconnect**: Pro+'ta nadir; background execution açıksa zaten
  devam eder. Yoksa son checkpoint'ten devam et.
- **`No module named X`**: ilk pip install hücresini tekrar çalıştır.

## Minimum koşu (zaman kısıtlıysa)

~1.5 saat A100'de:

1. Setup hücreleri
2. `prepare_data.sh`
3. GPT-2 baseline train + eval
4. T5 baseline train + eval
5. Son zip hücresi

Diffusion-LM ve ablation'ları ikinci oturumda. Pro+ background execution
ile bu ayrımı yapmaya gerek kalmayabilir — tek oturumda tüm pipeline
geçer.
