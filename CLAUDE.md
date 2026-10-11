# Proje: Çoklu Öğretmenli Damıtma ile Türkçe Embedding Modellerinin Geliştirilmesi

Yıldız Teknik Üniversitesi Bilgisayar Mühendisliği bitirme projesi (Ar-Ge türü).

- **Başlık (TR):** Çoklu Öğretmenli Damıtma ile Türkçe Embedding Modellerinin Geliştirilmesi
- **Başlık (EN):** (henüz belirlenmedi, `rapor/information.tex` içinde `\titleEN`)
- **Danışman:** Muzaffer Kaan YÜCE
- **Konu:** Birden fazla öğretmen modelden bilgi damıtma (multi-teacher knowledge distillation) ile Türkçe için daha iyi metin embedding modelleri geliştirmek.

## Dizin yapısı

Tek git reposu: kod, rapor (LaTeX) ve araştırma notları birlikte tutulur. Uzak repo: https://github.com/mericberktas/bitirme (ana dal `main`; Colab notebook'u bu adresten klonlar).

- `src/turkish_distill/` — Python paketi: `config`, `losses`, `models`, `teachers` (shard cache), `data`, `train`, `encode_teachers`, `prepare_corpus`, `checkpoint`, `logging_utils`, `diagnostics`, `eval/sts`, `utils`.
- `configs/` — YAML deney tanımları (`single_teacher_qwen3_8b.yaml`, `multi_teacher_a.yaml`).
- `notebooks/colab_pipeline.ipynb` — ince Colab notebook'u (Drive bağla, repoyu klonla, `pip install -e .`, corpus → öğretmen cache → eğitim → TensorBoard).
- `notebooks/colab_eval.ipynb` — yalnızca değerlendirme: model preset'leri (sabit revision'larla), TR-MTEB koşusu, sonuçların karşılaştırılması. Eğitimden ayrı runtime'da çalıştırılır (GPU çakışması; `mteb_tr` `datasets<3` sabitler).
- `src/turkish_distill/eval/trmteb.py` — parametrik TR-MTEB değerlendirmesi (`mteb_tr` çatalı, `MTEB(Turkish)`, 26 görev). Varsayılan protokol talimatsız/önneksiz (TR-MTEB yazarlarının yöntemi); Mean(task) ve Mean(type), bitext dahil/hariç. Sonuçlar `bitirme/eval/<label>/`.
- `tests/` — CPU'da çalışan testler (`pytest`); küçük sahte backbone ve sahte öğretmenlerle. Gerçek model/GPU ile henüz denenmedi.
- `README.md`, `pyproject.toml`, `requirements.txt`, `.gitignore`.
- `arastirma/ogretmen_veri_benchmark_arastirmasi.md` — öğretmen adayları, veri setleri, benchmark'lar, damıtma literatürü ve skor tabloları (kaynaklı, belirsiz yerler işaretli).
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
- Kaynakları `references.bib` içine ekle ve `\cite{}` ile at; uydurma kaynak veya sayı yazma.
- Şablon yönerge metinlerini (ör. "Giriş bölümü ... içermelidir") gerçek içerik yazıldığında kaldır.
- Deney sonuçlarını ve sayıları yalnızca gerçekten elde edilen çıktılardan raporla.
