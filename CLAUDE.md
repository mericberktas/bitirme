# Proje: Çoklu Öğretmenli Damıtma ile Türkçe Embedding Modellerinin Geliştirilmesi

Yıldız Teknik Üniversitesi Bilgisayar Mühendisliği bitirme projesi (Ar-Ge türü).

- **Başlık (TR):** Çoklu Öğretmenli Damıtma ile Türkçe Embedding Modellerinin Geliştirilmesi
- **Başlık (EN):** (henüz belirlenmedi, `rapor/information.tex` içinde `\titleEN`)
- **Danışman:** Muzaffer Kaan YÜCE
- **Konu:** Birden fazla öğretmen modelden bilgi damıtma (multi-teacher knowledge distillation) ile Türkçe için daha iyi metin embedding modelleri geliştirmek.

## Dizin yapısı

Git reposu yalnızca kodu (paket, config, notebook, test) tutar. `rapor/` (LaTeX) ve `arastirma/` (notlar) 2026-10-17 itibarıyla `.gitignore`'da ve takipten çıkarıldı: yalnızca yerelde durur (geçmiş commit'lerde eski halleri vardır). Uzak repo: https://github.com/mericberktas/bitirme (ana dal `main`; Colab notebook'u bu adresten klonlar).

- `src/turkish_distill/` — Python paketi: `config`, `losses`, `models`, `teachers` (shard cache), `data`, `train`, `encode_teachers`, `prepare_corpus`, `prepare_pairs` (üç QA setini `(soru, pasaj)` çiftlerine birleştirir, dekontamine eder), `benchmark_texts` (TR-MTEB test verisini HF'den indirir, metinlerini dışa aktarır), `contrastive_config` / `contrastive` (InfoNCE loss, GradCache adımı, tekrarsız batch örnekleyici, doğrulama retrieval metriği) / `train_contrastive` (contrastive baseline döngüsü ve CLI), `checkpoint`, `logging_utils`, `diagnostics`, `eval/sts`, `utils`.
- `data/` — yerel, **repoya girmez** (`.gitignore`): `data/raw/` indirilen ham QA setleri (Wikipedia-QA, TurHistQuAD, WikiRAG-TR; ~630 MB), `data/benchmark/` TR-MTEB test verisi (HF'den, `manifest.json` commit hash'leri; ~280 MB) ve `data/benchmark_texts.jsonl` (2.036.771 benzersiz metin), `data/stage1/` `prepare_pairs` çıktıları (`pairs_train/val.jsonl`, `corpus_train.jsonl`, `stats.json`; dekontamine edilmiş). Aşama 1 verisi henüz Drive'a yüklenmedi.
- `configs/` — YAML deney tanımları (`single_teacher_qwen3_8b.yaml`, `multi_teacher_a.yaml`, `contrastive_mnrl.yaml`).
- `notebooks/colab_pipeline.ipynb` — ince Colab notebook'u (Drive bağla, repoyu klonla, `pip install -e .`, corpus → öğretmen cache → eğitim → TensorBoard).
- `notebooks/colab_contrastive.ipynb` — contrastive baseline notebook'u; parametreler (öğrenci modeli, loss, scale, batch, chunk, lr, epoch) form değişkenleri, koşu klasörü adı ayarları taşır; sonunda koşuları doğrulama MRR@10'a göre karşılaştırır. Girdi: Drive'da `data/stage1/pairs_*.jsonl`.
- `notebooks/colab_eval.ipynb` — yalnızca değerlendirme: model preset'leri (sabit revision'larla), TR-MTEB koşusu, sonuçların karşılaştırılması. Eğitimden ayrı runtime'da çalıştırılır (GPU çakışması; `mteb_tr` `datasets<3` sabitler).
- `src/turkish_distill/eval/trmteb.py` — parametrik TR-MTEB değerlendirmesi (`mteb_tr` çatalı, `MTEB(Turkish)`, 26 görev). Varsayılan protokol talimatsız/önneksiz (TR-MTEB yazarlarının yöntemi); Mean(task) ve Mean(type), bitext dahil/hariç; ayrıca bizim tanımladığımız 9 beceri grubunun ortalaması (`TASK_GROUPS`, `summarize` çıktısında ikinci tablo). Sonuçlar `bitirme/eval/<label>/`.
- `tests/` — CPU'da çalışan testler (`pytest`); küçük sahte backbone ve sahte öğretmenlerle. Gerçek model/GPU ile henüz denenmedi.
- `README.md`, `pyproject.toml`, `requirements.txt`, `.gitignore`.
- `arastirma/ogretmen_veri_benchmark_arastirmasi.md` — öğretmen adayları, veri setleri, benchmark'lar, damıtma literatürü ve skor tabloları (kaynaklı, belirsiz yerler işaretli). Bölüm 11: çoklu öğretmenli damıtma literatür taraması (kanıt, olumsuz sonuçlar, öğretmen seçimi ve birleştirme yöntemleri, bizim için çıkarımlar); metindeki `anahtar` adları `rapor/references.bib` girdileriyle birebir eşleşir.
- `rapor/` — LaTeX proje kitabı (YTÜ şablonu). Ar-Ge türü kullanılıyor.
  - `main.tex` — ana dosya; `\projecttype` Ar-Ge için `1` olmalı (şu an `0`).
  - `information.tex` — öğrenci/başlık/özet/anahtar kelime/kısaltma bilgileri (şu an şablon örnek metni).
  - `RD_projectChapters/1-6-chapter.tex` — kullanılacak bölümler (Giriş, Ön İnceleme, Sistem Analizi, Fizibilite, Materyal ve Veri Kümesi, Metot, Performans Analizi, Karşılaştırmalı Analiz).
  - `SW_projectChapters/` — yazılım türü bölümleri, bu projede kullanılmıyor.
  - `references.bib` — kaynakça (biber/biblatex, IEEE stili).
  - `ytucestyle.sty`, `frontPages.tex`, `cv.tex` — şablon dosyaları, gerekmedikçe dokunma.
  - `Article_Template/`, `Project_Template/`, `LatexTutorial/` — yardımcı şablon materyalleri.

## Teknik kararlar

- **Öğrenci model:** `ytu-ce-cosmos/modernbert-tr-base-1k` (150M, 768 boyut, 1024 bağlam). Aynı lab'ın tek öğretmenli `modernbert-tr-embed` modeli (Qwen3-8B, MTEB(Turkish) 68,14) farklı bir tabanla (`modernbert-tr-base`, 8K) eğitilmiş; yayımlanmış harici referans olarak kullanılır.
- **Deney tasarımı:** Skor kıyası kontrollü yapılır: aynı öğrenci/veri/bütçe ile (1) lab'ın tek öğretmenli tarifinin kendi yeniden üretimi, (2) çoklu öğretmenli yöntem; ayrıca `modernbert-tr-1k-embed` aynı protokolle ölçülür. Yayımlanmış sayılar yalnızca "farklı taban" notuyla referans verilir.
- **Danışman kararı (2026-10-14):** Önce baseline'lar: aynı öğrenci (`modernbert-tr-base-1k`) ile (1) farklı loss'larla contrastive eğitim (ör. `CachedMultipleNegativesRankingLoss`; veri ve formatımıza uygun loss'lar araştırılacak), (2) tek öğretmenli damıtma, (3) iki öğretmenli, farklı ağırlıklı damıtma. Veri olarak danışmanın önerdiği üç QA seti (Wikipedia-QA, TurHistQuAD, WikiRAG-TR). Plan: bu "aşama 1" ile yöntem seçilir (seçim doğrulama kümesiyle, TR-MTEB test skoruyla değil), ana eğitim sonra Cosmos korpusu (+ aşama 1 metinleri) ile yapılır. Öğretmen cache'i şu an tüm korpus parmak iziyle anahtarlı; Cosmos ile birleştirmek için parça (chunk) tabanlı cache gerekir (yapılmadı).
- **Aşama 1 verisi (ölçülen):** Wikipedia-QA (CC-BY-NC-SA, 4.006.987 LLM üretimi soru-cevap, makale başına ort. 9), TurHistQuAD (MIT, insan yazımı, 14.221 eğitim satırı; 126'sı SQuAD kökenli ve atılıyor), WikiRAG-TR (Apache-2.0, 5.725 pozitif). `prepare_pairs` kuralları: Wikipedia-QA'dan 100-2.000 karakterlik makaleler, makale başına en fazla 3 soru, 25.000 çift; pasaj başına en fazla 10 soru (TurHistQuAD'da bir pasajda 118 soru vardı); doğrulama pasajları eğitimden çıkarılır. **Bulgu:** TurHistQuAD'ın test bölümü fiilen TR-MTEB'in TQuAD testi (1.330 çiftin 892'si örtüşüyor); dekontaminasyon doğrulama kümesinin %49'unu ve eğitimden 286 çifti attı. Son durum: 38.706 eğitim çifti (TurHistQuAD 8.553, Wikipedia-QA 24.939, WikiRAG-TR 5.214), 937 doğrulama çifti (437 TurHistQuAD + 500 WikiRAG-TR), damıtma için 70.136 metin.
- **Contrastive baseline:** Veri yalnızca pozitif (soru, pasaj) çifti olduğundan loss ailesi MNRL/InfoNCE (`mnrl`, `mnrl_symmetric`; scale 20 veya 50). Büyük batch için GradCache (`train.chunk_size`), matematiksel olarak düz adımla aynı (testle doğrulandı). Aynı pasajın soruları aynı batch'e girmez. Model seçimi doğrulama retrieval metriğiyle (MRR@10, recall@k), TR-MTEB test skoruyla değil. CoSENT şimdilik yok (skorlu çift yok), zor negatif madenciliği yok. Gerçek GPU koşusu henüz yapılmadı; hiperparametreler (lr 3e-5, 3 epoch, scale 20) ayarlanmamış başlangıç değerleridir.
- **Yerel Python:** Testler Python 3.13 (`C:/Users/meric/AppData/Local/Programs/Python/Python313/python.exe`; torch, pytest, pyarrow, transformers kurulu) ile çalışır: `PYTHONPATH=src python -m pytest -q tests`. Cihazda yer yok, yeni venv kurma. Global Python 3.14'te torch yok.
- **Öğretmen seçimi (ölçüme dayalı):** Qwen3-8B (67,96) > Qwen3-4B (66,03, fark >1 puan; bu yüzden elendi). Görev bazlı oracle tavanı: Qwen8B+BGE-M3+mE5 için 69,7 (en iyi tek öğretmene +1,8); BGE-M3 NLI/duygu'da, Qwen zor retrieval/kümelemede, mE5 STS/haber sınıflandırmada güçlü. İki öğretmenli çift için öneri: Qwen3-8B + BGE-M3.
- **Birincil benchmark:** TR-MTEB (sürümü sabitle; bitext dahil ve hariç ortalama ver). İkincil: Mizan, TabiBench IR+STS, STSb-TR/All-NLI-TR. Test setleri eğitim verisinden dekontamine edilir.
- **Ortam:** Eğitim ve değerlendirme tamamen Google Colab'da. Kod `src/` altında bir Python paketi; notebook'lar ince (clone + `pip install -e .` + config + çağrı). Deneyler YAML config ile tanımlanır. Bağımlılık sürümleri sabitlenir.
- **Model indirme:** Hugging Face modelleri her oturumda Colab'ın **yerel diskine** iner (`HF_HOME=/content/hf_cache`). HF cache'i Drive'a konmaz: sembolik bağlantılar ve büyük dosyalar Drive'da güvenilir saklanmıyor, Qwen3-8B ağırlıkları bozuk yazıldı (79 baytlık dosyalar, `SafetensorError: header too large`). Drive'a yalnızca öğretmenlerin ürettiği vektörler (cache), corpus, checkpoint ve loglar gider.
- **Öğretmen kodlaması:** Shard'lar `data.shard_size` metinlik parçalar (varsayılan 10.000; Qwen3-8B için L4'te ölçülen ~156 sn/1000 metin ile shard başına ~26 dk). Oturum kopunca en fazla bir shard kaybedilir. `shard_size` bir cache oluştuktan sonra değiştirilemez; kod uyuşmazlığı ve bozuk shard'ı hata olarak yakalar. Ölçülen Qwen3-8B hızı: 200K metin ≈ 8,7 saat, tam 978K ≈ 42 saat.
- **Değerlendirme (kendi ölçümlerimiz, L4, TR-MTEB talimatsız protokol):** Qwen3-8B 67,96; mE5-instruct 67,51 (sorgu talimatıyla; talimatsız 65,86); BGE-M3 67,20; lab'ın `modernbert-tr-embed` modeli 67,03; ham `modernbert-tr-base-1k` 48,83. Önek/talimat yalnızca retrieval skorunu değiştirir; `*_instr` koşuları `reuse_from` ile düz koşunun diğer 15 görev sonucunu kopyalar.
- **Sürüm sabitleme:** Config'te her öğretmen ve öğrenci için `revision` (HF commit hash) sabitlenir; öğretmen cache'inin `meta.json`'ına yazılır ve cache açılırken kontrol edilir (2026-10-11 itibarıyla hash'ler HF API'sinden alındı).
- **Depolama:** Her şey (öğretmen embedding cache'i, checkpoint'ler, loglar) Google Drive'da (5 TB). Cache az sayıda büyük shard olarak tutulur ve eğitimden önce yerel diske kopyalanır. Checkpoint önce yerel diske yazılıp Drive'a atomik taşınır; `resume` desteği zorunlu.
- **İzleme:** TensorBoard. Ayrıca her zaman `metrics.jsonl` yazılır (gerçek kaynak). Loglananlar: kayıp bileşenleri (öğretmen başına), LR, grad normu, throughput; periyodik olarak öğrenci-öğretmen cosine'ı, embedding geometrisi (ortalama çiftli cosine, efektif rank), hızlı STS/retrieval değerlendirmesi; çalıştırma başına config, git SHA, seed, `pip freeze`, veri/cache sürümü.
- **Drive düzeni:** `bitirme/cache/teachers/<öğretmen>/<veri_sürümü>/`, `bitirme/runs/<deney>/<run_id>/{config.yaml,metrics.jsonl,checkpoints/,eval/}`.
- **Sırlar:** HF token vb. Colab Secrets ile; repoya veya Drive config'lerine yazılmaz. Gemma gated model, token ve lisans onayı gerekir.

## Çalışma kuralları

- Rapor tamamen İngilizce yazılacak; akademik ve resmi bir üslup kullan. Şablon seçeneği `computer` Türkçe, `senior` İngilizce başlıklar üretiyor; `main.tex` içindeki `\usepackage[computer, final]{ytucestyle}` satırı buna göre gözden geçirilmeli.
- Kullanıcıyla sohbet Türkçe yürütülür.
- Kaynakları `references.bib` içine ekle ve `\cite{}` ile at; uydurma kaynak veya sayı yazma. `references.bib` içinde şablonun örnek girdilerinin altında, bu projede okunan ~55 makalenin girdileri var (anahtar: ilk yazar soyadı + yıl + kısa ad; arXiv'e ilk yüklenme yılı değil, konferans yılı). arXiv-only girdiler `@misc`; ACL Anthology/NeurIPS resmî BibTeX'leri olduğu gibi alındı. Şablonun eski örnek girdileri (`WinNT`, `guo2004learning` vb.) ilgisiz, rapor yazılırken silinebilir.
- Şablon yönerge metinlerini (ör. "Giriş bölümü ... içermelidir") gerçek içerik yazıldığında kaldır.
- Deney sonuçlarını ve sayıları yalnızca gerçekten elde edilen çıktılardan raporla.
- Override değerlerinde `3e-05` biçimi PyYAML'de metin olur; `parse_override_value` bunu sayıya çevirir (iki config yükleyicisi de kullanır).
