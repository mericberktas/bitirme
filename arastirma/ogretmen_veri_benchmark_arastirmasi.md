# Araştırma Notları: Öğretmen Modeller, Veri Setleri ve Türkçe Benchmark'lar

**Proje:** Çoklu Öğretmenli Damıtma ile Türkçe Embedding Modellerinin Geliştirilmesi
**Öğrenci (student) model:** [`ytu-ce-cosmos/modernbert-tr-base-1k`](https://huggingface.co/ytu-ce-cosmos/modernbert-tr-base-1k)
**Araştırma tarihi:** 2026-10-09

> **Okuma notu.** Bu dosyadaki her sayı bir kaynağa bağlıdır. Doğrulayamadığım veya kaynaklar arasında çeliştiği yerleri **[DOĞRULANMADI]** veya **[ÇELİŞKİ]** diye işaretledim. Farklı makalelerin skorları farklı protokollerle (farklı görev alt kümesi, farklı ortalama yöntemi) hesaplandığı için tablolar arası doğrudan kıyas yapmayın. Seçim yapmadan önce adayları kendi ortamınızda aynı protokolle yeniden ölçmenizi öneririm (bkz. Bölüm 5).

## İçindekiler

0. Öğrenci model ve yakın çalışmalar
1. Öğretmen model adayları
2. Veri setleri ve kendimiz veri üretme seçenekleri
3. Türkçe benchmark'lar
4. Öneriler
5. Açık sorular ve sonraki adımlar
6. Türkçe'de damıtma örnekleri (ayrıntılı)
7. İngilizce / dilden bağımsız çoklu öğretmenli damıtma
8. Japonca ve diğer sondan eklemeli dillerde damıtma
9. Sentez: bizim tasarım kararları için çıkarımlar
10. Skor tabloları (ModernBERT-TR ailesi ve tüm damıtma çalışmaları)
11. Kaynaklar

---

## 0. Öğrenci model ve yakın çalışmalar

### 0.1 Öğrenci model: ModernBERT-TR-base-1k

Model kartından ([kaynak](https://huggingface.co/ytu-ce-cosmos/modernbert-tr-base-1k)):

| Özellik | Değer |
|---|---|
| Mimari | ModernBERT: 22 katman, 768 hidden, 12 head, RoPE, GLU, alternatif local-global attention, Flash Attention |
| Parametre | ~150M |
| Bağlam uzunluğu | 1.024 token |
| Tokenizer | 50K kelime dağarcıklı WordPiece (Türkçe) |
| Ön eğitim verisi | ~144,4B token (FineWeb-2 Türkçe 41,2B + BertTurk Corpus x5 31,0B, 2 epoch) |
| Lisans | Apache-2.0 |
| Kendi raporladığı sonuç | Dondurulmuş linear probing (11 Türkçe görev): 60,2 ortalama; TabiBench tam fine-tuning: 77,28 |

**Önemli:** Bu model bir *MLM encoder*; hazır bir cümle-embedding modeli değil. Yani damıtma sırasında embedding uzayını sıfırdan öğrenecek (pooling + olası projeksiyon katmanı). Başka bir ModernBERT-TR ailesi modeli için bu durumun nasıl sonuçlandığı aşağıdaki "yakın çalışmalar"da var.

### 0.2 Doğrudan ilgili çalışmalar (rakip / referans)

| Çalışma | Ne yaptı | Öğretmen(ler) | Sonuç |
|---|---|---|---|
| **MoganBert-Embed** ([arXiv 2608.25768](https://arxiv.org/abs/2608.25768)) | 149M'lik ModernBERT tabanlı encoder'dan 2 fazlı embedding: (1) teacher distillation, (2) multi-signal contrastive fine-tuning | **Tek öğretmen:** Qwen3-Embedding-8B | TR-MTEB (26 görev) ortalaması **68,30**; öğretmenin 68,66'sının %99,5'i. Retrieval zayıf nokta (59,58 vs öğretmen 64,46). Yazarlar sonuçları "ön çalışma" olarak niteliyor. |
| **ModernBERT-TR-Embed** ([model kartı](https://huggingface.co/ytu-ce-cosmos/modernbert-tr-embed)) | **Aynı laboratuvarın** damıtılmış embedding modeli: `modernbert-tr-base` + 7,9M pasajla Qwen3-8B damıtması (Jasper/Stella tarifi) + 3 ince ayar + ağırlık ortalaması | **Tek öğretmen:** Qwen3-Embedding-8B | MTEB(Turkish) ortalaması **68,14** (Bölüm 10.1.1). **Bizim doğrudan baseline'ımız.** |
| **embeddingmagibu-200m** ([arXiv 2605.29992](https://arxiv.org/abs/2605.29992)) | Tokenizer cerrahisi + offline damıtma, cosine loss | **Tek öğretmen:** EmbeddingGemma-300M | TR-MTEB ort. 63,9 (öğretmen 65,2), STSb-TR'de öğretmeni geçti (77,55 vs 73,84 Pearson). Eğitim: 1 epoch, tek A100, ~4 saat. |
| **TurkEmbed / TurkEmbed4Retrieval** ([2511.08376](https://arxiv.org/abs/2511.08376), [2511.07595](https://arxiv.org/abs/2511.07595)) | Multilingual encoder'ları NLI + STS (ve retrieval) ile Türkçe'ye fine-tune | Damıtma yok (denetimli) | STSb-TR'de ~0,845 Pearson |
| **TR-MTEB modelleri** ([EMNLP Findings 2025](https://aclanthology.org/2025.findings-emnlp.471/)) | BERTurk üzerinde 34,2M çiftle contrastive ön eğitim + denetimli fine-tune | Damıtma yok | `turkish-embedding-model-fine-tuned` (110M) |
| **Mecellem / Mursit** ([arXiv 2601.16018](https://arxiv.org/abs/2601.16018)) | Hukuk alanı için ModernBERT ve Qwen3 CPT, retrieval modelleri | Damıtma yok (InfoNCE ve Qwen3-InfoNCE) | Mizan'da Mursit-Large-TR-Retrieval 56,43 |

**Araştırma boşluğu (özgünlük iddiası için kritik):** Aradığım kadarıyla **Türkçe için çoklu (multi-teacher) embedding damıtma** yayımlanmış değil: MoganBert-Embed ve embeddingmagibu tek öğretmenli. Ancak çoklu öğretmenli damıtma başka dillerde var (İngilizce: Jasper & Stella, Formont ve ark.; Japonca: GLuCoSE v2). Bu yüzden iddia "çoklu öğretmenli damıtma ilk kez" değil, **"Türkçe için ilk" ve "öğretmen sayısı / seçimi / ağırlıklandırmasının sistematik analizi"** olmalı. Ayrıntılar Bölüm 6-9'da. Bu iddiayı yazmadan önce son aylardaki yeni arXiv çıktılarını bir kez daha taramak gerekir; alan hızlı ilerliyor.

---

## 1. Öğretmen model adayları

### 1.1 Seçim kriterleri

1. **Türkçe başarı** (TR-MTEB, Mizan, TurkEmbed çalışması gibi bağımsız sonuçlar).
2. **Lisans / kullanım şartı:** damıtılmış modeli yayımlayabiliyor muyuz? (ticari olmayan, "competing model" yasakları, Gemma "Model Derivative" maddesi.)
3. **Tamamlayıcılık:** Öğretmenler birbirinin zayıf olduğu yerde güçlü olmalı, yoksa "çoklu" olmanın anlamı kalmaz.
4. **Pratiklik:** parametre sayısı, çıkarım maliyeti (damıtma için milyonlarca metin kodlanacak), embedding boyutu.
5. **Boyut uyumu:** Öğrenci 768 boyutlu. Öğretmen çıktıları 768 / 1024 / 4096; her öğretmen için ayrı projeksiyon başlığı veya ilişkisel (benzerlik matrisi) kayıp gerekir.

### 1.2 Türkçe sonuçlara bakış

**A) TR-MTEB (26 görev), MoganBert-Embed makalesindeki tablodan** ([kaynak](https://arxiv.org/abs/2608.25768)). Kategori ortalamaları:

| Model | Param | Bağlam | Genel | Retrieval | Pair Clf. | STS | Clf. | Clust. | Bitext |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3-Embedding-8B | 7,57B | 40.960 | **68,66** | **64,46** | 62,76 | 80,04 | 72,95 | 63,01 | 98,18 |
| multilingual-e5-large-instruct | 560M | 512 | 67,47 | 61,05 | 63,03 | **81,23** | 73,70 | 61,87 | **98,99** |
| BGE-M3 | 568M | 8192 | 67,36 | 60,26 | **68,72** | 79,60 | 72,79 | 60,80 | **98,99** |
| Mursit-Large-TR | 404M | 2048 | 62,00 | 55,70 | 55,19 | 74,60 | 68,66 | 61,53 | 86,72 |
| trmteb-tr-embed | 111M | 512 | 59,71 | 52,54 | 56,81 | 74,96 | 70,62 | 63,17 | 37,75 |

**Gözlem:** En iyi öğretmen kategori bazında değişiyor: retrieval'da Qwen3-8B, STS'de e5-large-instruct, pair classification'da BGE-M3. Çoklu öğretmen fikrini destekleyen somut kanıt bu.

**B) Mizan Türk leaderboard'u** ([canlı sayfa](https://newmindai-mizan.hf.space/); 52 model, 14 Türkçe görev, "MTEB Score" kategori ortalaması). İlk sıralar:

| Sıra | Model | MTEB Skoru | Legal Skoru | Param |
|---|---|---|---|---|
| 1 | google/embeddinggemma-300m | 65,42 | 50,63 | 307M |
| 2 | newmindai/bge-m3-stsb | 63,53 | 49,97 | 567M |
| 3 | BAAI/bge-m3 | 62,87 | 51,16 | 567M |
| 4 | Lajavaness/bilingual-embedding-large | 62,47 | 47,36 | 559M |
| 5 | newmindai/TurkEmbed4STS | 62,42 | 47,23 | 305M |
| 6 | intfloat/multilingual-e5-large | 61,51 | 46,95 | 559M |
| 8 | ytu-ce-cosmos/turkish-e5-large | 60,36 | 48,20 | 559M |
| 9 | Alibaba-NLP/gte-multilingual-base | 60,12 | 46,49 | 305M |
| 10 | nomic-ai/nomic-embed-text-v2-moe | 59,54 | 50,75 | 475M |
| 13 | intfloat/multilingual-e5-large-instruct | 58,85 | 46,69 | 559M |
| 23 | nvidia/llama-embed-nemotron-8b | 53,52 | 32,55 | 8B |
| 24 | KaLM-embedding-multilingual-mini-instruct-v2.5 | 52,84 | 32,62 | 494M |

- **[ÇELİŞKİ]** Mizan ile TR-MTEB sıralamaları uyuşmuyor (ör. e5-large-instruct TR-MTEB'de üst sırada, Mizan'da 13.). Sebep büyük olasılıkla farklı görev kümeleri (Mizan 14 görev + hukuk). Hangisinin hedefimize daha yakın olduğuna karar vermeliyiz.
- Mizan'da **Qwen3-Embedding listede yok** (Mursit-Embed-Qwen3 CPT modelleri ayrı). Yani Qwen3'ün Türkçe skoru için bağımsız tek kaynak MoganBert makalesi.
- Sayfadaki iki satır (KaLM'nin ikinci bir kopyası, sıra 10) tutarsız render edilmiş; onu dışarıda bıraktım.

**C) TurkEmbed makalesindeki STS/NLI sonuçları** ([kaynak](https://arxiv.org/abs/2511.08376)): Qwen3-Embedding-8B, bge-m3 ve gte-multilingual-base STSb-TR'de 0,795-0,804 Pearson aralığında; multilingual-e5-large-instruct 0,846 ile en yüksek. STS22-TR'de Qwen3-8B 0,701 ile önde. Yani STS'de de tek bir "kazanan" yok.

### 1.3 Aday kartları (artı / eksi)

#### 1) Qwen3-Embedding-8B (ve 4B / 0.6B kardeşleri) — **güçlü aday**
[Kaynak](https://huggingface.co/Qwen/Qwen3-Embedding-8B)

- **Özellikler:** ~8B param, 4096 boyut (MRL ile 32-4096), 32K bağlam, 100+ dil, **Apache-2.0**, talimat duyarlı (instruction-aware). MMTEB çoklu dil ortalaması 70,58 (8B), 69,45 (4B), 64,33 (0.6B).
- **Artılar:**
  - Lisans en rahat grupta, damıtılmış modeli yayımlamak sorunsuz.
  - TR-MTEB'de en yüksek ortalama (68,66) ve en iyi retrieval (64,46).
  - Türkçe'de **önceki çalışmada kanıtlanmış öğretmen** (MoganBert-Embed 99,5% oranında yakaladı).
  - 4B ve 0.6B kardeşleri, maliyet-kalite ödünleşimi için aynı aileden seçenek sunuyor.
- **Eksiler:**
  - Çıkarım maliyeti yüksek (8B), büyük korpusu kodlamak GPU saatleri tutar. Offline (önceden hesaplanmış) damıtma şart.
  - 4096 boyut vs 768 öğrenci: projeksiyon gerekir (MoganBert tersine, öğrenciyi 3072'ye yükseltti).
  - Mizan'da bağımsız doğrulama yok; Türkçe skoru tek makaleye dayanıyor.
  - Decoder tabanlı; son-token pooling ve talimat formatı hatalı kullanılırsa performans düşer.
  - MoganBert-Embed zaten bunu yaptığı için, tek başına özgünlük katmaz.

#### 2) EmbeddingGemma-300M — **güçlü aday (verimli)**
[Kaynak](https://huggingface.co/google/embeddinggemma-300m)

- **Özellikler:** ~300M param, **768 boyut (öğrenciyle aynı)**, 2048 bağlam, 100+ dil, MRL (768/512/256/128), MTEB çoklu dil v2 ortalaması 61,15. Lisans: **Gemma Terms of Use**.
- **Artılar:**
  - Mizan'da **1. sıra** (65,42), Türkçe hukuk skorlarında da iyi.
  - 768 boyut öğrenciyle birebir uyumlu, projeksiyona gerek yok.
  - Küçük ve hızlı, tüm eğitim verisini kodlamak ucuz.
  - Türkçe'de embeddingmagibu'nun öğretmeni olarak denenmiş.
- **Eksiler:**
  - **Lisans riski:** Gemma şartları "Model Derivative" tanımına açıkça *"distillation methods that use intermediate data representations"* ve *"generation of synthetic data Outputs by Gemma"* yöntemlerini dahil ediyor ([şartlar](https://ai.google.dev/gemma/terms)). Çıktılar kendi başına Model Derivative sayılmıyor, ama damıtma yoluyla elde edilen model sayılabilir. Dağıtırsak Gemma'nın kullanım kısıtlamalarını ve şartlar kopyasını aktarmamız gerekir. **Hukuki yorum yapmadım; yayımlamadan önce kontrol edilmeli.**
  - Float16 desteklemiyor (float32 / bfloat16 gerekli).
  - TR-MTEB'de Qwen3'ün gerisinde (65,2 vs 68,66; farklı protokol).
  - Önek formatı (`task: ... | query: ...`) zorunlu, yanlış kullanım performansı bozar.

#### 3) BGE-M3 — **güçlü aday**
[Kaynak](https://huggingface.co/BAAI/bge-m3)

- **Özellikler:** XLM-R large tabanlı, ~568M, 1024 boyut, 8192 bağlam, 100+ dil, **MIT**. Dense + sparse + multi-vector (ColBERT tarzı) çıktı.
- **Artılar:**
  - Hem TR-MTEB (67,36) hem Mizan'da (62,87, #3) tutarlı üst sıralarda.
  - **Pair classification'da en iyi** (68,72), bitext'te zirve.
  - MIT lisansı, çok rahat.
  - Dense + sparse + ColBERT çıktıları farklı damıtma sinyalleri sunabilir (ör. token düzeyinde hizalama).
  - 8192 bağlam; öğrenci 1024 ile sınırlı olsa da uzun belgeler parçalanarak kodlanabilir.
- **Eksiler:**
  - 1024 boyut, projeksiyon gerekir.
  - XLM-R tokenizer'ı Türkçe'yi öğrencinin 50K Türkçe WordPiece'ine göre farklı böler; token düzeyinde hizalama imkânsız, yalnızca cümle düzeyinde damıtma yapılabilir.
  - Çok dilli olduğundan Türkçe'ye özgü ince semantik farklarda sınırlı kalabilir.

#### 4) multilingual-e5-large-instruct — **STS / pair için tamamlayıcı**
[Kaynak](https://huggingface.co/intfloat/multilingual-e5-large-instruct)

- **Özellikler:** XLM-R large tabanlı (24 katman), 1024 boyut, **512 token bağlam**, **MIT**, sorgular için `Instruct: ...\nQuery: ...` formatı.
- **Artılar:**
  - TR-MTEB'de **en iyi STS (81,23)**, STSb-TR'de de en yüksek (0,846/0,854).
  - Retrieval'da (61,05) hâlâ iyi.
  - MIT, rahat lisans; hafif sayılabilecek bir "büyük" model (560M).
- **Eksiler:**
  - 512 token, uzun metinleri keser (öğrenci 1024 bağlama sahip; uzun örnekler için bu öğretmen kullanılamaz).
  - Kosinüs skorları 0,7-1,0 bandına sıkışıyor (düşük InfoNCE sıcaklığı), kayıp fonksiyonu seçerken dikkat (yalnızca göreli sıralama anlamlı).
  - Talimat verilmezse performans düşer.
  - Mizan'da 13. sırada; görev kümesine duyarlı.

#### 5) ytu-ce-cosmos/turkish-e5-large — **Türkçe'ye özel, aynı laboratuvar**
[Model](https://huggingface.co/ytu-ce-cosmos/turkish-e5-large)

- **Özellikler:** ~559M. Arama sonuçlarına göre `multilingual-e5-large-instruct` temelli, MIT lisanslı, Türkçe retrieval için fine-tune edilmiş **[DOĞRULANMADI: model kartını doğrudan okuyamadım; eğitim verisi bilinmiyor]**.
- **Artılar:**
  - Öğrenci modelle aynı grubun (ytu-ce-cosmos) modeli: eğitim ayrıntılarına ulaşmak muhtemelen kolay.
  - Mizan'da 8. (60,36), hukuk skoru 48,20 (temel e5-large-instruct'tan 1,5 puan iyi).
  - **MTEB(Turkish) ortalaması 67,17** (Retr. 61,5; Clf. 72,6; Pair 62,8; Kümeleme 60,9; STS 80,0; Bitext 99,3), multilingual-e5-large'ın (66,56) üstünde. Kaynak: `modernbert-tr-embed` model kartının karşılaştırma tablosu (Bölüm 10.1.1). Bu sayı, Mizan'daki sıralamanın tersine Türkçe fine-tune'un genel kazancını destekliyor.
- **Eksiler:**
  - Mizan'da temel multilingual-e5-large'ın (61,51) altında; Türkçe fine-tune'un genel kazancı belirsiz.
  - Eğitim verisi belgelenmemiş (en azından özetlerde) → sızıntı (contamination) riskini bilemiyoruz.
  - Öğretmen olarak e5-large-instruct ile büyük ölçüde örtüşecektir (çeşitlilik düşük).

#### 6) gte-multilingual-base
[Kaynak](https://huggingface.co/Alibaba-NLP/gte-multilingual-base)

- **Özellikler:** ~305M, 768 boyut (128-768 arası ayarlanabilir), 8192 bağlam, 70+ dil, **Apache-2.0**, encoder-only, sparse çıktı da veriyor.
- **Artılar:** Küçük ve hızlı, **768 boyut öğrenciyle uyumlu**, uzun bağlam, rahat lisans. Mizan'da 9. (60,12).
- **Eksiler:** TR-MTEB ve Mizan'da en üstlerin altında (TR-MTEB makale tablosu okunurken sütunlar kaydığı için kesin skoru yazmadım); bağımsız olarak diğerlerinden belirgin biçimde üstün olduğu bir kategori yok.

#### 7) Türkçe fine-tune'lu NLI/STS uzmanları (TurkEmbed4STS, bge-m3-stsb)
[Mizan'daki kayıtlar](https://newmindai-mizan.hf.space/), [TurkEmbed](https://arxiv.org/abs/2511.08376)

- **Artılar:** STS/NLI için Türkçe uzmanlaştırılmış; Mizan'da bge-m3-stsb 2., TurkEmbed4STS 5. Çoklu öğretmen kurulumunda "STS uzmanı" rolü verilebilir.
- **Eksiler:** Veriler **makine çevirisi** NLI/STS (All-NLI-TR, STSb-TR); çeviri hatalarını öğrenciye aktarma riski. Lisansları ve model kartlarını bu çalışmada doğrulamadım **[DOĞRULANMADI]**. Taban modelleriyle (gte, bge-m3) yüksek oranda örtüşür.

#### 8) nomic-embed-text-v2-moe
- **Özellikler:** ~475M, MoE, 2048 bağlam (Mizan'daki değer).
- **Artılar:** Mizan'da hukuk skoru en yüksek (50,75), farklı mimari → çeşitlilik sağlar. TurkEmbed çalışmasında STSb-TR'de 0,828/0,834.
- **Eksiler:** Lisansı bu araştırmada doğrulanmadı **[DOĞRULANMADI]**; Mizan MTEB skoru ortalama (59,54, 10.); TR-MTEB'de sonucu yok.

#### 9) Kullanımı kısıtlı / tavsiye edilmeyen adaylar

| Model | Neden zor |
|---|---|
| **llama-embed-nemotron-8b** ([kart](https://huggingface.co/nvidia/llama-embed-nemotron-8b)) | **Yalnızca ticari olmayan / araştırma amaçlı** (NVIDIA lisansı + Llama 3.1 lisansı). MMTEB çoklu dilde Borda 1. (69,46), ama Mizan'da yalnızca 53,52 (23. sıra) → Türkçe'de beklentinin altında. |
| **jina-embeddings-v3** ([jina.ai](https://jina.ai/models/jina-embeddings-v3/)) | **CC-BY-NC-4.0** (ticari olmayan); 570M; kendi sayfasında "yeni modellerle değiştirildi" uyarısı; Türkçe desteği doğrulanamadı. |
| **gemini-embedding-001** (API) | MMTEB ortalaması 68,32 ile güçlü, ancak API; Gemini API ek şartları "rakip model geliştirme" yasağı içeriyor (arşivlenmiş sürümler üzerinden görüldü; embedding'e özgü bir madde bulamadım). Ücretli, tekrar üretilemezlik ve maliyet. |
| **KaLM-embedding-multilingual-mini-instruct-v2.5** | Mizan'da 52,84 (24.), TurkEmbed'de STS22-TR'de zayıf (0,342). |
| **snowflake-arctic-embed-l-v2.0** ([kart](https://huggingface.co/Snowflake/snowflake-arctic-embed-l-v2.0)) | Apache-2.0, 74 dil; ancak Mizan/TR-MTEB'de sonucu bulunamadı, Türkçe etkinliği **[DOĞRULANMADI]**. Ölçmeye değer bir aday. |
| **IBM granite-embedding-278m-multilingual** | Mizan'da 53,93 (21.), öğretmen olarak yeterince güçlü değil. |

### 1.4 Önerilen öğretmen kümeleri (hipotez, ölçümle doğrulanmalı)

| Küme | Üyeler | Gerekçe | Risk |
|---|---|---|---|
| **A (güvenli, açık lisans)** | Qwen3-Embedding-8B + BGE-M3 + multilingual-e5-large-instruct | Hepsi Apache-2.0/MIT; retrieval (Qwen3), pair (BGE-M3), STS (e5) uzmanlıkları birbirini tamamlıyor | 4096/1024/1024 boyut farkı → projeksiyon başlıkları; e5'in 512 bağlamı |
| **B (verimli)** | EmbeddingGemma-300M + gte-multilingual-base + BGE-M3 | Hepsi ≤568M, hızlı kodlama; ikisi 768 boyutlu | Gemma lisansı (distillation = Model Derivative) |
| **C (geniş)** | A + EmbeddingGemma-300M | Mizan 1.'si de dahil; 4 farklı ailenin ortak oyu | Gemma lisansı; kodlama maliyeti |

Önerim: **A ile başlayıp** (lisans en temiz), EmbeddingGemma'yı "ablation" olarak eklemek. Ayrıca "tek öğretmen" ve "çoklu öğretmen" arasında kıyas, projenin ana deney tasarımı olabilir (MoganBert-Embed'in Qwen3 tek-öğretmen sonucu doğal bir referans noktası).

---

## 2. Veri setleri ve kendimiz veri üretme

### 2.1 Kilit gözlem: damıtma etiketsiz metinle çalışır

Offline damıtmada öğretmenler metni kodlar, öğrenci bu vektörleri taklit eder. **Etiket gerekmez.** MoganBert Faz 1'de ve embeddingmagibu'da böyle yapılmış (embeddingmagibu: yalnızca ~580K satırlık çok dilli Wikipedia kümesi, 1 epoch, ~4 saat). Etiketli çift gerektiren aşama yalnızca isteğe bağlı sonraki contrastive fine-tune (MoganBert Faz 2). Dolayısıyla veri sorunu, etiket sorunundan çok **kapsama / alan çeşitliliği / temizlik / lisans** sorunu.

### 2.2 Etiketsiz Türkçe metin havuzları (damıtma için ana yakıt)

| Veri seti | İçerik / boyut | Lisans | Not |
|---|---|---|---|
| [Cosmos-Turkish-Corpus-v1.0](https://huggingface.co/datasets/ytu-ce-cosmos/Cosmos-Turkish-Corpus-v1.0) | ~15B token, 9,08M satır; haber, forum, Wikipedia | **CC-BY-4.0** | Aynı lab. URL tabanlı tekilleştirme yapılmış, içerik düzeyinde ek filtreleme önerilmiş |
| [FineWeb-2 (Türkçe)](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) | ~95M Türkçe metin (derleme listesine göre) | **[DOĞRULANMADI]** (ODC-By olduğunu biliyorum, bu araştırmada doğrulamadım) | Öğrenci model bununla ön eğitilmiş → alan uyumu yüksek |
| [FinePDFs](https://huggingface.co/datasets/HuggingFaceFW/finepdfs) | ~1,7M Türkçe PDF metni | [DOĞRULANMADI] | Daha "akademik/resmî" register |
| HPLT 2.0 cleaned (tur_Latn), vngrs-web-corpus, C4/mC4 | Web taramaları | Her birine bakılmalı | Gürültü yüksek, filtre gerekir |
| Türkçe Wikipedia / `news-tr-1.8M` / `wikisource` | Ansiklopedik, haber | Çoğu CC-BY-SA / çeşitli | TR-MTEB eğitim derlemesinde kullanılmış kaynaklar |

([Kaynak listesi: awesome-turkish-language-models](https://github.com/kesimeg/awesome-turkish-language-models))

### 2.3 Hazır çiftli / etiketli Türkçe veri setleri

| Veri seti | Tür | Boyut | Kalite / lisans notu |
|---|---|---|---|
| **TR-MTEB 34,2M süzülmüş çift derlemesi** ([makale](https://aclanthology.org/2025.findings-emnlp.471/), HF org [trmteb](https://huggingface.co/trmteb)) | Zayıf denetimli cümle çiftleri (50+ kaynak: mqa, mfaq, tapaco, Wikipedia QA, MLSUM, vb.) | 62,5M çiftten süzülmüş 34,2M (e5 tabanlı model, benzerlik eşiği 0,4) | Makale: **yalnızca araştırma / ticari olmayan** kullanım. Tam adı: HF'de `trmteb/turkish_embedding_model_training_data` (61M) ve temizlenmiş sürümü görünüyor **[DOĞRULANMADI: tam repo adını sayfada gördüm, indirmedim]** |
| **nli_tr** (SNLI-TR + MNLI-TR) ([boun-tabi/nli_tr](https://huggingface.co/datasets/boun-tabi/nli_tr)) | NLI | SNLI-TR 550K eğitim | Amazon Translate ile makine çevirisi. Lisans **[DOĞRULANMADI]** |
| **STSb-TR** ([emrecan/stsb-mt-turkish](https://huggingface.co/datasets/emrecan/stsb-mt-turkish)) | Anlamsal benzerlik | ~5,75K eğitim | Azure ile makine çevirisi, uzman kontrolü yok. Test kümesi aynı zamanda TR-MTEB'in STS testi → **eğitimde test bölümünü kullanmayın** |
| **TrGLUE** ([turkish-nlp-suite/TrGLUE](https://huggingface.co/datasets/turkish-nlp-suite/TrGLUE)) | GLUE tarzı: TrMRPC 5,18K, TrSTS-B 3,06K, TrQQP 369K, TrMNLI 202K, TrQNLI 140K, TrRTE 5,78K | | Sentence-pair alt kümeleri paraphrase/NLI için kullanılabilir |
| **MS MARCO-TR** ([parsak/msmarco-tr](https://huggingface.co/datasets/parsak/msmarco-tr)) | Passage retrieval (718K pasaj, 533K qrel, 603K sorgu, 1M triple) | ~2,85M satır | Apache-2.0 etiketli ama **veri kartı boş**; gözümle baktığım örneklerde kesik pasajlar, çevrilmemiş İngilizce kalıntıları, bozuk cümleler ve neredeyse aynı pasajlar var. Mutlaka filtrelenmeli |
| **TQuAD, XQuAD-tr, SQuAD-TR** | Okuduğunu anlama / retrieval | TQuAD 892 sorgu; XQuAD 1,19K; SQuAD-TR ~9K | Küçük ve aynı zamanda **TR-MTEB test kümeleri** → eğitimden uzak tutun |
| **WikiRAG-TR**, mfaq, mqa, tapaco, MedTurkQuAD, Türkçe hukuk NLI vb. | Çeşitli QA/paraphrase | TR-MTEB Ek Tablo 10'da boyutlarıyla | Alan çeşitliliği iyi, kalite değişken |
| **Türkçe paraphrase derlemesi (~800K)** ([ICNLSP 2022](https://aclanthology.org/2022.icnlsp-1.14.pdf)) | Paraphrase | ~800K | HF'de erişilebilirliğini **doğrulamadım** |
| **FineTranslations** | TR-EN paralel | ~58M çift (derleme listesine göre) | Çok dilli/bitext hizalama için; öğrenci tek dilli olacaksa gereksiz |

**Eksik / bulunamayanlar:**
- **mMARCO Türkçe** için sonuç bulamadım **[DOĞRULANMADI]**; TR-MTEB MS MARCO-TR'yi kullanıyor (Kesgin, Yüce ve Amasyalı 2023 atfıyla; yazarlardan biri danışmanımız Muzaffer Kaan Yüce, kökeni ve kalitesini ona sormak yararlı olabilir).
- **Native (çevrilmemiş) Türkçe retrieval eğitim verisi çok az.** Native olanlar çoğunlukla TQuAD (küçük), haber/özet çiftleri ve forum verileri.

### 2.4 Kendimiz veri üretebilir miyiz? — Evet, ve muhtemelen yapmalıyız

Öncelik sırasına göre seçenekler:

1. **Doğal çift madenciliği (ucuz, native, lisansı en temiz)**
   Başlık-gövde, manşet-ilk paragraf, Wikipedia bölüm başlığı-bölüm, haber başlığı-özet, forum soru-cevap çiftleri. Cosmos korpusu (CC-BY-4.0) bunun için hazır ham madde. TR-MTEB derlemesinin yaptığı şeyin kendimize uyarlanmış, daha küçük ve daha temiz sürümü.

2. **LLM ile sentetik sorgu üretimi** (doc2query / InPars / Promptagator mantığı)
   Gerçek Türkçe pasajlardan Türkçe sorgu, paraphrase ve "zor negatif" üretmek. Üretici LLM olarak Qwen3 veya Gemma 3 tipi açık modeller düşünülebilir. **Önemli kısıtlar:**
   - Üretici modelin lisansı (özellikle Gemma: sentetik çıktılar da "Model Derivative" tanımına giriyor).
   - Kalite denetimi: MoganBert'te kullanılan 40.737 duygu-karşıtlık üçlüsü LLM ile üretilmiş ama insan doğrulaması yok; bu tür verinin riskini yazar da not etmiş.
   - Benchmark sızıntısını önlemek için test kümesi pasajlarını üretim havuzuna sokmayın.

3. **İngilizce setlerin LLM ile çevrilmesi + LLM-hakem filtresi**
   TR-MTEB'in izlediği yöntem: Cetvel'e göre en iyi çeviri modeli aya-expanse-35B, hesap kısıtı yüzünden 8B sürümü kullanılmış; FAIL işaretli örnekler daha büyük modelle yeniden çevrilmiş; LLM-hakem insanla %85,2 uyum sağlamış (115 örnek, küçük bir kalibrasyon). Bu yöntem Türkçe MS MARCO, NQ, HotpotQA, AllNLI vb. için tekrarlanabilir. **Eksi:** çeviri hataları, "translationese", ve aya-expanse'in CC-BY-NC lisansı **[DOĞRULANMADI, ticari olmayan olduğunu hatırlıyorum]**.

4. **Öğretmenle zayıf etiketleme (pseudo-labeling)**
   Rastgele/yakın cümle çiftlerini topluluğun öğretmenleriyle puanlayıp STS tarzı yumuşak etiket üretmek. Zaten damıtmanın doğal uzantısı: öğretmen vektörleri/benzerlikleri, etiketin kendisi.

5. **Geri çeviri (back-translation) ile paraphrase üretimi**: ucuz ama çeviri yönü yan etkileri.

**Hukuki / etik not:** CC-BY-4.0 (Cosmos) atıf gerektirir; TR-MTEB derlemesi ve bazı çeviri setleri ticari olmayan kullanım şartı taşıyor. Hangi kaynağı, hangi lisansla, nihai modele nasıl taşıyacağımızı bir **veri-lisans tablosu** olarak yazılı tutmalıyız (rapora da girer).

### 2.5 Veri hakkında öneri

- **Ana yakıt:** Cosmos korpusu + FineWeb-2 Türkçe'den örneklenmiş, tekilleştirilmiş, uzunluğa göre dilimlenmiş (≤1024 token) 1-5M metin; öğretmenlerle bir kez kodlayıp diske kaydet.
- **Görev dengeleme:** Kısa cümleler (STS/NLI tarzı), orta pasajlar (retrieval), uzun metinler (haber/akademik) için ayrı kotalar.
- **Zor örnek zenginleştirme:** Retrieval için öğretmenlerle hard-negative madenciliği; NLI/STS için TrGLUE + nli_tr.
- **Test ayrımı:** TR-MTEB ve TabiBench'in tüm test kümelerini ve bunların kaynak pasajlarını eğitim havuzundan çıkaran bir **dekontaminasyon betiği** yazmalıyız (n-gram / hash eşleştirme).

---

## 3. Türkçe benchmark'lar

### 3.1 TR-MTEB — birincil embedding benchmark'ı
[Makale](https://aclanthology.org/2025.findings-emnlp.471/) · [HF org](https://huggingface.co/trmteb) · değerlendirme kodu `selmanbaysan/mteb_tr` (ikincil kaynaktan)

- **Kapsam:** 26 veri seti, makalede 6 görev türü: sınıflandırma (8), kümeleme (2), pair classification (3), bitext mining (1), STS (1), retrieval (11). Metrikler: accuracy, V-measure, average precision, F1, Spearman, nDCG@10.
- **Kaynaklar:** Yerli Türkçe veri (haber, THY duygu, ironi, hakaret tespiti), önceden çevrilmiş (MNLI-TR, SNLI-TR, STSb-TR, SQuAD-TR, MS MARCO-TR), XNLI/XQuAD Türkçe bölümü, ve **bu çalışmada çevrilmiş 7 BEIR seti** (NFCorpus, ArguAna, CQADupStack-Gaming, FiQA, Quora, SCIDOCS, SciFact).
- **Artılar:** Türkçe için en kapsamlı ve en çok kullanılan embedding benchmark'ı; kolay erişilebilir; yakın tarihli tüm rakip çalışmalar (MoganBert, magibu, Mursit) bununla kıyas yapıyor → karşılaştırılabilirlik.
- **Eksiler:**
  - Büyük bölümü **makine çevirisi** (translationese, artefaktlar); yazarlar da bunu sınırlılık olarak belirtiyor.
  - **Bitext mining** tek dilli öğrenci için anlamsız: yazarların kendi tek dilli modeli 37,7 aldı, çok dilli modeller ~99. Bitext'i ayrı raporlayıp ortalamayı hem dahil hem hariç vermek dürüst olur.
  - Sürüm kayması: sonraki bir makale (7 kategori; reranking, NLI ayrı) farklı bir dağılım tarif ediyor **[ÇELİŞKİ]**; hangi sürümle çalıştığımızı sabitleyip raporlamalıyız.
  - Bu benchmark'ın bazı retrieval testleri küçük (ör. TQuAD 892 sorgu).
  - Eğitim derlemesiyle (34,2M çift) aynı ekipten çıktığı için, o derlemeyi kullanırsak sızıntı kontrolü şart.

### 3.2 Mizan — canlı leaderboard (görünürlük için)
[Sayfa](https://huggingface.co/spaces/newmindai/Mizan)

- **Kapsam:** 52 model; "MTEB Score" 14 Türkçe görevin (sınıflandırma, retrieval, pair classification, kümeleme, STS) kategori ortalaması; ayrıca "Legal Score" (sözleşme, mevzuat, içtihat retrieval: TurkishLegalQA, TurkishTaxRulings, TurkishCourtOfCassation); tokenizer kalite sütunları.
- **Artılar:** Güncel sıralama ve bağımsız bir platform; model göndermek mümkün (HF kimlik doğrulama, yönetici incelemesi, model halka açık olmalı) → **sonucumuzu yayımlamak için hedef.** ModernBERT-TR tabanlı Mursit modelleri de burada → doğrudan rakip.
- **Eksiler:** Görev listesinin ayrıntısı sayfada tam görünmüyor, MTEB skoru TR-MTEB'den farklı bir alt küme; hukuk skoru bizim hedefimizin dışında; sıralamalar TR-MTEB ile çelişiyor.

### 3.3 TabiBench — encoder'lar için geniş NLU + retrieval paketi
[TabiBERT makalesi](https://arxiv.org/abs/2512.23065)

- **Kapsam:** 27 veri seti, 8 kategori: metin sınıflandırma (4), token sınıflandırma, STS, NLI (2), QA (2), **bilgi erişimi (6)**, **kod erişimi (4)**, akademik anlama (3). IR setleri TR-MTEB'in derlediği altı veriyi içeriyor. Kod ve akademik setler GPT-4.1 ile çevrilmiş.
- **Artılar:** Öğrenci modelin kendi raporladığı benchmark (ModernBERT-TR kartı 77,28 veriyor) → doğrudan kıyas imkânı. Standartlaştırılmış fine-tuning protokolü, 8 kategori.
- **Eksiler:** Embedding'den çok *fine-tune edilmiş encoder* değerlendirmesi; yalnızca IR/STS alt kümesi bizim için anlamlı. Aynı yazarlar hem benchmark'ı hem modeli yaptı (makale bağımsızlık önlemlerini anlatıyor ama yine de potansiyel yanlılık). Model kartı 28 görev, makale (v4) 27 veri seti diyor **[ÇELİŞKİ, küçük]**. Kod ve akademik setler çeviri.

### 3.4 TR-TEB
[ResearchGate kaydı](https://www.researchgate.net/publication/407438143_TR-TEB_Turkish_Text_Embedding_Benchmark)

- 5 görev kategorisi (sınıflandırma, pair classification, kümeleme, retrieval, STS). Arama özetine göre BGE-M3 63,76, mE5-large 62,61, mE5-base 60,39.
- **[DOĞRULANMADI]** Yalnızca bir özet görebildim; veri seti listesi, erişim ve lisansı bilmiyorum. TR-MTEB ile ilişkisi belirsiz (eş zamanlı ayrı bir çalışma gibi görünüyor). Ayrıntıya girmeden önce yazarları ve bağlantıyı doğrulamak gerekir.

### 3.5 Daha küçük / tamamlayıcı değerlendirmeler

| Benchmark | Ne ölçer | Not |
|---|---|---|
| **STSb-TR, All-NLI-TR, STS22-TR** | Cümle benzerliği ve NLI | TurkEmbed çalışmasının raporladığı üçlü; STSb-TR TR-MTEB'in de STS görevi. Literatürle kıyas için kolay |
| **TurkColBERT'in 5 Türkçe BEIR seti** ([blog](https://huggingface.co/blog/nmmursit/late-interaction-models)) | SciFact-TR, ArguAna-TR, FiQA-TR, Scidocs-TR, NFCorpus-TR (zero-shot retrieval) | TR-MTEB'in retrieval alt kümesi. Yazarlar veri setlerinin küçük/kısmen çeviri ve sonuçların tek koşu olduğunu söylüyor |
| **MMTEB** ([Enevoldsen ve ark. 2025](https://arxiv.org/abs/2502.13595)) | 500+ görev, 250+ dil | Türkçe görev sayısı sınırlı (TR-MTEB makalesine göre başlangıçta 2 sınıflandırma + 1 retrieval); uluslararası görünürlük için uygun ama Türkçe'ye özgü derinlik yok |
| **TrGLUE / SentiTurca** ([arXiv 2512.22100](https://arxiv.org/html/2512.22100v1)) | Türkçe GLUE benzeri NLU + duygu | Embedding değil; öğrenci modelin alt görev transferini göstermek için isteğe bağlı |
| **Cetvel** (Kesen ve ark. 2024; TR-MTEB makalesinde atıf var, bağlantısını doğrulamadım) | Türkçe LLM değerlendirmesi | Embedding değil; **sentetik veri/çeviri için LLM seçerken** kullanılabilir (TR-MTEB'in yaptığı gibi) |
| **MTEB-Turkish (Mecellem)** ([arXiv 2601.16018](https://arxiv.org/abs/2601.16018)) | MTEB tarzı + hukuk retrieval | Hukuk alanı odaklı; Mizan'ın "Legal Score"u bunun uzantısı |
| **ModernBERT-TR kartındaki 11 görevlik linear probing** | Dondurulmuş temsil kalitesi | Öğrenci taban modelin başlangıç çizgisi (60,2); damıtma sonrası aynı protokolle yeniden ölçmek "temsil iyileşti mi" sorusuna temiz cevap verir |

### 3.6 Benchmark önerisi

1. **Birincil:** TR-MTEB (sürümü sabitleyerek; genel ortalamayı bitext dahil/hariç iki biçimde ve kategori kırılımıyla).
2. **İkincil:** Mizan (görünürlük ve rakip kıyas), TabiBench IR + STS alt kümesi.
3. **Literatür karşılaştırması:** STSb-TR + All-NLI-TR (TurkEmbed ile aynı koşul).
4. **Referans satırları:** öğretmenlerin kendisi (tavan), tek öğretmenli damıtma (MoganBert tarzı), damıtmasız fine-tune, ModernBERT-TR-base-1k ham hali.
5. **Dekontaminasyon:** test kümesi pasajları eğitim havuzundan çıkarılmış olmalı.

---

## 4. Öneriler (kısa)

1. **Öğretmenler:** Qwen3-Embedding-8B (retrieval) + BGE-M3 (pair/genel) + multilingual-e5-large-instruct (STS). Hepsi açık lisanslı ve TR-MTEB'de birbirini tamamlıyor. EmbeddingGemma-300M'i lisans netleştikten sonra ablation olarak ekle.
2. **Yöntem:** Offline damıtma; her öğretmen için ayrı projeksiyon başlığı veya ilişkisel (benzerlik matrisi) kayıp; Jasper & Stella tarzı üçlü kayıp; öğretmen ağırlıklandırma (ör. görev türüne göre) ve çelişen öğretmenleri dengeleme ana araştırma sorusu olabilir.
3. **Veri:** Etiketsiz Cosmos + FineWeb-2 ana yakıt; doğal çift madenciliği ve LLM sentetik sorgular ikinci aşama (contrastive) için.
4. **Değerlendirme:** TR-MTEB birincil, Mizan'a gönderim; tek vs çoklu öğretmen ablation'ı raporun omurgası.
5. **Rapor için kritik:** Veri/model lisans tablosu ve dekontaminasyon prosedürü.

## 5. Açık sorular ve sonraki adımlar

- [ ] Adayları (Qwen3-8B, Qwen3-4B, BGE-M3, e5-large-instruct, EmbeddingGemma, gte-multilingual, turkish-e5-large, arctic-embed-l-v2) **aynı protokolle TR-MTEB üzerinde kendimiz ölçelim**; literatürdeki sıralamalar çelişiyor.
- [ ] Öğretmenlerin hata/uyum analizi: hangi örneklerde birbirinden ayrılıyorlar? (Çoklu öğretmenin gerçek katkısı burada.)
- [ ] Gemma Terms için kurum/danışman düzeyinde hukuki yorum.
- [ ] `trmteb` 34,2M derlemesinin tam repo adı, lisansı ve test kümeleriyle örtüşme kontrolü.
- [ ] `turkish-e5-large` model kartını okuyup eğitim verisini öğrenmek (aynı lab).
- [ ] TR-TEB'in gerçekten ayrı bir benchmark olduğunu ve erişilebilirliğini doğrulamak.
- [ ] Cetvel makalesinin doğru bağlantısı (yukarıda placeholder bıraktım).
- [ ] nomic-embed-text-v2-moe, TurkEmbed4STS, bge-m3-stsb lisanslarını doğrulamak.
- [ ] Danışman Muzaffer Kaan Yüce'ye sormak: MS MARCO-TR'nin kökeni/kalitesi, laboratuvarın elindeki Türkçe çift veri setleri.

## 6. Türkçe'de damıtma örnekleri (ayrıntılı)

> Aşağıdaki 2026 tarihli arXiv çalışmaları (2605.xxxxx, 2608.xxxxx) çok yeni ve hakem değerlendirmesinden geçmemiş ön baskılar. Yazarların kendileri bazı sonuçları "ön çalışma" olarak niteliyor. Makalelerin düz metnine HTML/PDF üzerinden ulaştım; tabloların tamamını okuyamadığım yerleri belirttim.

**Özet tablo**

| Çalışma | Ne damıtıyor | Öğretmen | Sayı | Kayıp |
|---|---|---|---|---|
| MoganBert-Embed (2608.25768) | Cümle embedding'i | Qwen3-Embedding-8B | 1 | Cosine (+ GOR düzenleyici), sonra InfoNCE + damıtma |
| embeddingmagibu-200m (2605.29992) | Cümle embedding'i | EmbeddingGemma-300M | 1 | Cosine |
| MoganColBERT-TR (2608.26344) | Skor dağılımı (late interaction) | bge-reranker-v2-m3 (cross-encoder) | 1 | KL ıraksaması |
| DistilBERTurk | Dil modeli (genel amaçlı) | BERTurk | 1 | Standart DistilBERT |
| TurkEmbed, TR-MTEB modelleri, Mursit | Damıtma **yok** (denetimli/contrastive) | - | - | - |

**Sonuç: Türkçe'de yayımlanmış tüm embedding damıtma örnekleri tek öğretmenli.** Hiçbiri öğretmenleri birbiriyle karşılaştırmıyor.

### 6.1 MoganBert-Embed ([arXiv 2608.25768](https://arxiv.org/abs/2608.25768))

Bizim projeye en yakın çalışma: Türkçe'de sıfırdan eğitilmiş 149M'lik ModernBERT tabanlı bir encoder'ı embedding modeline çeviriyor.

- **Faz 1 (damıtma):** Öğretmen Qwen3-Embedding-8B (3072 boyutlu çıktı), öğrenci 768 → 3072 boyutuna **projekte ediliyor**; öğretmen kırpılmıyor. Kayıp: `1,0·L_distill + 0,05·L_GOR`; damıtma terimi cosine tabanlı, GOR (global orthogonal regularization) 768 boyutlu backbone çıktısına uygulanıyor. Öğretmene görev talimatı **verilmiyor**; öğrenci öğretmenle **aynı girdi formatını** görmeli, yoksa öğrenilen eşleme kayıyor.
- **Neden gerekli:** Ham MLM temsilleri aşırı anizotropik (ortalama cosine `cos_raw` = 0,9841). Faz 1 sonrasında `cos_raw` 0,0851'e düştü, efektif rank 117,9 → 157,2, sıfır-atış retrieval (10 set, nDCG@10) **0,2361 → 0,5927** oldu. Yani damıtma burada *embedding uzayını ilk kez kuran* adım.
- **Faz 2 (contrastive):** InfoNCE (τ = 0,02, hard negative'li) + damıtma terimi korunuyor; damıtma ağırlığı 2,0'dan 0,25'e düşürülüyor (sıfırlanmıyor). STS için CoSENT, NLI için contradiction'ı hard negative olarak kullanan kayıplar. Birkaç kontrol noktası "soup" (ağırlık ortalaması) ile birleştirilmiş.
- **Faz 1 tek başına yetmiyor:** Makaledeki bir satıra göre Faz 1 tek başına MTEB(Turkish) genel skoru **63,53**; Faz 2'nin v3 sürümü 66,97. Tablodaki nihai 68,30 muhtemelen soup sonucu (**bu eşlemeyi ben çıkardım, makalede açıkça görmedim**).
- **Sonuç:** TR-MTEB 26 görevde ortalama 68,30, öğretmenin (68,66) %99,5'i; retrieval zayıf (59,58 vs 64,46).
- **Raporlanmayanlar:** Faz 1 korpusunun boyutu/kaynağı, adım sayısı, batch size, hesaplama; hiçbir "tek vs çoklu öğretmen" deneyi yok.
- **Önemli not:** 40.737 duygu-karşıtlık üçlüsü LLM üretimi ve insan doğrulamasız.

### 6.2 embeddingmagibu-200m ([arXiv 2605.29992](https://arxiv.org/abs/2605.29992))

Bir *çok dilli* embedding modelini Türkçe'ye uyarlamak: tokenizer cerrahisi + offline damıtma.

- **Tokenizer:** 131.072'lik hibrit sözlük: Cosmos Türkçe Korpus'tan en sık 64K Türkçe token + Wikipedia-40-lang'dan seçilmiş çok dilli tokenlar. Yeni token embedding'i = öğretmenin ilgili alt-sözcüklerinin ortalaması (ağırlık korunarak klonlama).
- **Damıtma:** Öğretmen EmbeddingGemma-300M, embedding'ler **bir kez önceden hesaplanıyor**; öğrenci cosine benzerlik kaybıyla eğitiliyor.
- **Veri:** ~580K satırlık dengeli Wikipedia-40-lang kümesi (Türkçe ve İngilizce 100K'şar, diğer 38 dil 10K'şar).
- **Kurulum:** 1 epoch, batch 256, LR 5e-5, bf16, tek A100 80GB, ~4 saat, 5-20 dolar. **Çok ucuz.**
- **Sonuç:** STSbTR'de Pearson 77,55 (öğretmen 73,84; öğrenci öğretmeni geçti), TR-MTEB 26 görevde ortalama 63,9 (öğretmen 65,2; 26 modelden 7.). Retrieval 72,2 (öğretmen 75,9), sınıflandırma 68,5 (71,8), STS 77,5 (72,9), NLI 67,9 (60,6).
- **Yazarların sınırlılıkları:** Öğrenci öğretmenin hatalarını ve önyargılarını miras alıyor; sözlüğü 256K'dan 128K'ya indirmek düşük kaynaklı diller için zayıflatabilir; doğrusal pooling/projeksiyon; çok uzun belgeler test edilmedi; **Türkçe NLI/IR üzerinde denetimli contrastive eğitim yok.**
- **Bizim için ders:** (a) etiketsiz küçük bir veriyle bile (580K) öğretmeni STS/NLI'da yakalamak/geçmek mümkün; (b) tokenizer farkı sorunu burada *öğrencinin tokenizer'ı değiştirilerek* çözülmüş. Bizde öğrencinin tokenizer'ı sabit (Türkçe WordPiece 50K), bu avantaj: sözcük dağarcığı zaten Türkçe'ye uygun.

### 6.3 MoganColBERT-TR ([arXiv 2608.26344](https://arxiv.org/abs/2608.26344))

ModernBERT tabanlı Türkçe encoder'dan ColBERT tarzı çoklu vektörlü retrieval modeli (768 → 128 projeksiyon, MaxSim).

- **Damıtma:** Cross-encoder öğretmen `bge-reranker-v2-m3`'ün 1 pozitif + 7 mined negatif üzerindeki yumuşak skorları; kayıp PyLate'in distillation hedefi (öğrencinin MaxSim skor dağılımı ile öğretmeninki arasında KL). 1 epoch.
- **Yazarların sınırlılıkları:** Tek öğretmen kullanılmış; öğretmenin Türkçe'deki yanlılıkları öğrenciye geçmiş olabilir; **distillation vs contrastive için ablation yok.**
- **Bizim için ders:** Türkçe'de bge-reranker-v2-m3 öğretmen olarak denenmiş; retrieval için embedding öğretmenlerine eklenebilecek farklı *tür* bir öğretmen (skor tabanlı).

### 6.4 Diğer Türkçe örnekler

- **DistilBERTurk:** BERTurk'ten damıtılmış 67M'lik genel amaçlı dil modeli (embedding değil). Türkçe prosedürel dil anlama çalışmasında reranker olarak tam BERTurk'ten kötü sonuç verdi (R@1'de düşüş). Eski nesil, görev-bağımsız LM damıtması.
- **Translation Aligned Sentence Embeddings ([arXiv 2311.09748](https://arxiv.org/abs/2311.09748)):** Encoder-decoder modeli önce çeviri çiftleriyle hizalayıp sonra küçük Türkçe veriyle cümle embedding'ine fine-tune eden iki aşamalı yöntem. Sınıf olarak damıtma değil; ayrıntılı sonuçlarına ulaşamadım **[DOĞRULANMADI]**.
- **TurkColBERT, TurkEmbed, TR-MTEB modelleri, Mursit/Mecellem:** Damıtma kullanmıyor (denetimli ya da contrastive). Bunlar damıtmasız referans noktaları olarak işe yarar.

---

## 7. İngilizce / dilden bağımsız çoklu öğretmenli damıtma

### 7.1 Jasper & Stella ([arXiv 2412.19048](https://arxiv.org/abs/2412.19048)) — en çok anılan çoklu öğretmenli örnek

- **Öğretmenler (donmuş):** v2 HTML'e göre NV-Embed-v2 (4096 boyut) ve stella_en_1.5B_v5 (8192 boyut). **[ÇELİŞKİ]** Arama özeti v1'de ikinci öğretmen olarak bge-en-icl'i veriyor; sürümler arasında öğretmen listesi değişmiş olabilir.
- **Öğrenci:** stella_en_1.5B_v5'ten başlatılıyor (öğretmenlerden biri). Toplam 1,9B (görsel encoder dahil).
- **Boyut sorunu çözümü:** Öğretmen vektörleri **birleştiriliyor** (4096 + 8192 = 12.288); öğrencinin bir FC katmanı bu boyuta eşleniyor. Sonradan FC2-FC4 ile Matryoshka tarzı boyut azaltma.
- **Üç kayıp:** `L = λ1·L_cos + λ2·L_sim + λ3·L_resim`
  - `L_cos`: öğrenci ve birleşik öğretmen vektörleri arasında cosine kaybı;
  - `L_sim`: öğrencinin ikili benzerlik matrisi (S·Sᵀ) ile öğretmeninkinin (T·Tᵀ) MSE'si;
  - `L_resim`: öğretmenin bir çifti diğerinden daha benzer sıraladığı durumlarda hinge kaybı (CoSENT'ten esinli).
  - Ağırlıklar λ1=10, λ2=200, λ3=20, margin=0,015.
- **Veri:** ~8 milyon metin: %80 fineweb-edu, %20 sentence-transformers/embedding-training-data. Yazarlar açıkça "denetimli veriye ihtiyacımız yok" diyor. Metinlerin ~%30'u 1-10 cümlelik parçalara bölünmüş, %0,08'inin kelimeleri karıştırılmış.
- **4 aşama:** (1) yalnızca FC1 eğit, (2) FC1 + son 3 encoder katmanı, (3) boyut azaltma, (4) görsel hizalama.
- **Sonuç:** MTEB (56 veri seti, 24 Aralık 2024): 71,54, 3. sıra; NV-Embed-v2 72,31 (7,85B), stella 71,19 (1,54B).
- **ÖNEMLİ:** Makalede **kayıp ablation'ı yok ve tek öğretmen vs çoklu öğretmen karşılaştırması yok.** Çoklu öğretmenin gerçek katkısı kanıtlanmamış.

### 7.2 Formont ve ark., NeurIPS 2025 ([arXiv 2510.18680](https://arxiv.org/abs/2510.18680)) — "majority vote" çoklu öğretmen çerçevesi

- **Fikir:** Öğrencinin herhangi bir görevde öğretmenlerin çoğunluğuyla aynı kararı vermesi hedefi. Bu, öğrenci ile öğretmen embedding'leri arasındaki karşılıklı bilgiyle sınırlanıyor; **göreve özgü etiket gerektirmeyen** bir kayıp çıkıyor.
- **Kayıp:** Her öğretmen için, öğrenci embedding'inden tahmin edilen ortalama ve kovaryansa sahip bir Gaussian çekirdek altında öğretmen embedding'inin negatif log-olabilirliği; öğretmenler üzerinden ortalama. Öğretmenin kendi boyutu `d_k` olduğundan **her öğretmen için ayrı çekirdek** var; bu doğal olarak boyut uyumsuzluğunu çözüyor (yazarlar bunu açıkça "boyut sorununun çözümü" olarak anlatmıyor, ben çıkarım yaptım).
- **Öğretmenler (metin):** Serbestçe indirilebilir 4 embedding modeli (433M-7B); SFR-Embeddings-R_2, MTEB'de diğerlerinden 10+ puan güçlü. Öğretmen adlarının tam listesi okuduğum içerikte yoktu.
- **Öğrenciler:** Snowflake xs/s/m/l (23M, 33M, 109M, 335M).
- **Veri:** Hugging Face'ten ~6M tekilleştirilmiş kayıt (SPECTER, T5, Amazon QA, IMDB, SNLI, Quora, AG News, MEDI, DAIL Emotion), 2 epoch, batch 16, V100. (Metinde "66 milyon" da geçiyor, muhtemelen render hatası.)
- **Değerlendirme:** MTEB'in 33 görevi (11 kümeleme, 10 STS, 12 sınıflandırma). Kazanç **esas olarak sınıflandırmada**; kümeleme ve STS'de ılımlı (yazarlar, hedefin gömme uzayını yapılandırmadığını söylüyor). 109M'lik öğrenci 335M'lik öğrenciyi geçmiş.
- **Okuma uyarısı:** Metinde "22 puanlık" bir kazanım iddiası var ama Tablo 1 bunu göstermiyor; sayıları bu yüzden kullanmadım. Dilin İngilizce olduğu çıkarımım, açık bir ifade yok.
- **Bizim için önemi:** "Öğretmenleri nasıl birleştiririz?" sorusuna *kavramsal* bir cevap (çoğunluk oyu); Jasper'ın birleştirme (concat) yaklaşımına alternatif.

### 7.3 LEAF ([arXiv 2509.12539](https://arxiv.org/abs/2509.12539)) — tek öğretmen, ama pratik dersler

- Kayıp yalnızca **ℓ2 mesafesi**; hard negative ve alaka etiketi gerekmiyor, küçük batch (32) yetiyor; öğretmen embedding'leri önbelleğe alınıyor.
- MiniLM-L6 (23M) öğrencisi, `arctic-embed-m-v1.5` (109M) öğretmenin BEIR skorunun %96,1'ine ulaşmış; **asimetrik kullanım** (dokümanlar büyük öğretmenle, sorgular küçük öğrenciyle) %97,7'ye çıkarıyor. MiniLM/TinyBERT/DistilBERT'in ek kayıplarının fayda getirmediğini bildiriyor.
- Yazarlar çok dillilik, **çoklu öğretmen** ve uzun bağlamı gelecek çalışma olarak belirtiyor.
- **Bizim için önemi:** Öğrenci ve öğretmen aynı uzayda hizalıysa asimetrik retrieval mümkün; basit ℓ2/cosine yeterli olabilir. Ayrıca "karmaşık kayıp şart değil" argümanı.

### 7.4 EmbeddingGemma ([arXiv 2509.20354](https://arxiv.org/abs/2509.20354))

- Gemini Embedding öğretmenden **embedding eşleştirme** (sorgu, pozitif ve hard negative pasajlar için ayrı terimler, eşit ağırlıklı) + NCE + "spread-out" düzenleyici + MRL.
- Gemma 3'ten encoder-decoder'a çevrilip UL2 ile ön eğitilen encoder: MTEB çok dilli ablation'ında encoder-decoder başlangıcı 60,4, decoder-only 59,7, rastgele 45,2. "Model souping" (karışım ağırlıkları Bayes optimizasyonuyla seçilen çalışmaların ortalaması) 61,2 vs bireysel 60,1-60,4.
- **Damıtmanın katkısını tek başına gösteren ablation yok.**
- **Bizim için önemi:** (a) damıtma + contrastive birlikte kullanımı standart pratik; (b) soup/model ortalaması küçük ama tutarlı kazanç; (c) Gemma lisansı açısından bu modelin kendi eğitimi bizim yeniden üretmemiz gereken bir şey değil, ama **öğretmen olarak** kullanırsak Bölüm 1'deki lisans notu geçerli.

### 7.5 Diğer referanslar

- **Reimers & Gurevych, EMNLP 2020 ([arXiv 2004.09813](https://arxiv.org/abs/2004.09813)):** Klasik çok dilli damıtma şablonu: öğrenci, kaynak cümleyi ve çevirisini öğretmenin kaynak-cümle vektörüne MSE ile eşler. 50+ dili kapsıyor (Türkçe ve diğer sondan eklemeli dillerin dahil olduğunu doğrulamadım **[DOĞRULANMADI]**). Sentence-Transformers'ın `EmbedDistillLoss`'u (boyut farkı varsa öğrenilebilir projeksiyonla) bu ailenin pratik uygulaması ([doküman](https://www.sbert.net/examples/training/distillation/README.html)).
- **bge-m3:** "Self-knowledge distillation" kullanıyor; ama bu *kendi* dense/sparse/multi-vector çıktılarının birbirini öğretmesi, başka modelden damıtma değil.
- **MoL (Mixture of Layers), Knowledge-Based Systems:** **Farklı tokenizer'lı** modeller arasında embedding damıtması; arama özetine göre LLM2Vec-Mistral-7B, Qwen3-0.6B ve BGE-M3 öğretmenlerinden TinyBERT öğrencilere. Sayfaya erişemedim (HTTP 403), ayrıntıları **[DOĞRULANMADI]**. Bizim durumumuz (farklı tokenizer'lar) için ilgili; makaleye erişmeye değer.

---

## 8. Japonca ve diğer sondan eklemeli dillerde damıtma

Japonca sondan eklemeli (agglutinative) bir dil olarak Türkçe ile morfolojik benzerlik taşıyor; sözcük sınırları belirsiz, tokenizer sorunları ortak.

### 8.1 GLuCoSE-base-ja-v2 (PKSHA) — **çoklu öğretmenli Japonca örnek**

[Model kartı](https://huggingface.co/pkshatech/GLuCoSE-base-ja-v2)

- **Adım 1 (damıtma):** Üç öğretmen: **E5-mistral, gte-Qwen2, multilingual-e5-large**.
- **Adım 2 (genel cümle embedding contrastive):** JSNLI, MNLI, PAWS-X, JSeM, Mr.TyDi'den üçlüler.
- **Adım 3 (arama odaklı contrastive, 2 aşama):** (i) sentetik auto-wiki-qa; (ii) JQaRA, MQA, Japonca Wikipedia Human Retrieval, Mr.TyDi, MIRACL, Quiz Works, Quiz No Mori.
- **Taban:** GLuCoSE-base-ja (LUKE-Japanese-base-lite'tan). **JMTEB ortalaması 72,23** (Retrieval 73,36, STS 82,96, Sınıflandırma 74,21, Reranking 93,01, Kümeleme 48,65, Pair Clf 62,37). Lisans Apache-2.0.
- **Sınırlılık:** Model kartı üç öğretmenin *nasıl birleştirildiğini* (concat mı, ortalama mı, ayrı kayıp mı), ağırlıkları, damıtma verisini ve kaybı anlatmıyor; bir ablation da yok. **Bu, bulabildiğim sondan eklemeli bir dilde çoklu öğretmenli embedding damıtmasının en doğrudan örneği.** Makale/blog var mı ayrıca bakılmalı **[DOĞRULANMADI]**.
- **Bizim için önemi:** Üç öğretmen yapısı (decoder LLM-tabanlı iki + encoder-tabanlı bir) bizim önerdiğimiz karışıma (Qwen3-8B + BGE-M3 + e5) çok benziyor: *farklı mimari ailelerinden* öğretmenler.

### 8.2 Ruri v1/v2 ([arXiv 2409.07737](https://arxiv.org/abs/2409.07737)) — cross-encoder damıtması + sentetik veri

- **Damıtma türü:** Embedding'den embedding'e değil; **kendi eğittikleri Japonca reranker'ın (Ruri-Reranker-large, cross-encoder)** skorlarına damıtma. Retrieval/QA verisinde öğrenci cosine skorları ile öğretmen skorları **min-max normalize edilip** damıtılıyor; NLI verisinde ise düz contrastive kayıp ("ayrıştırılmış" kayıp).
- **Veri:** Contrastive ön eğitim için ~88,7M çift (Wikipedia'nın çeşitli kesimleri, MQA, CC-News, Wiki Atomic Edits, sentetik veriler); **LLM ile üretilmiş** AutoWikiQA (Swallow-MX ve Nemotron-4 340B ile) ve AutoWikiNLI (Nemotron-4 340B; entailment/contradiction üretimi, contradiction önce üretilerek basit negasyonlar azaltılmış; reward model ile en düşük %20 atılmış). Hard negative'ler BM25 ile.
- **İnce ayar:** ~293K örnek (JSQuAD, JaQuAD, AutoWikiQA, Quiz, JQaRA, MIRACL, Mr.TyDi, NU-SNLI/NU-MNLI makine çevirisi, JaNLI). Toplam batch 512, 15 hard negative, 512 token.
- **Ablation'lar:** (a) Contrastive ön eğitim şart: retrieval 56,62 → 69,41 (small), 52,99 → 69,82 (base), 57,84 → 73,02 (large). (b) Sentetik retrieval verisi retrievalı belirgin artırıyor, STS ve sınıflandırmada hafif düşüş. (c) İki aşamalı reranker eğitimi tek aşamadan iyi; contrastive ön eğitimli taban reranker'ı da iyileştiriyor.
- **Yazarların notu:** "Damıtma eğitimi hafifçe kararsız hale getirdi; gerçekten gerekli mi açık soru." Ayrıca task-homogeneous batching (batch'te tek veri seti) kullanmışlar.
- **JMTEB (16 veri seti):** Ruri-large 73,31, base 71,91, small 71,53; multilingual-e5-large 70,90; OpenAI text-embedding-3-large 74,05. Yani ~70-111M'lik Japonca modeli, mE5-large'ı geçiyor.
- **Veri lisansı notu:** mMARCO'yu bilinçli olarak **kullanmamışlar** ("çeviri kalitesi ve ticari lisans sorunları"). Bu, MS MARCO-TR kullanırken dikkat etmemiz gereken bir uyarı.
- **Tokenizer:** 32.768'lik sözlük (v1/v2), **v3'te 100K'ya çıkarılmış** (aşağıya bakınız).

### 8.3 Ruri v3 ([model kartı](https://huggingface.co/cl-nagoya/ruri-v3-310m))

- **ModernBERT-Ja** üzerine (bizim öğrenciyle aynı mimari ailesi), **100K SentencePiece sözlüğü** (v1/v2'de 32K), 8192 bağlam, harici Japonca sözcük bölücü gerekmiyor. İnce ayar verisi ~632K satır. 310m için JMTEB **77,24** (Retrieval 81,89, STS 81,22, Sınıflandırma 78,66).
- Model kartı damıtma veya sentetik veri **belirtmiyor** (yani bilmiyoruz; kartta olmaması kullanılmadığı anlamına gelmez).
- **Bizim için önemi:** ModernBERT mimarisi + dile özgü büyük sözlük ile 70M-310M ölçeğinde güçlü sondan eklemeli dil embedding modeli yapılabildiğinin kanıtı. Bizim öğrenci 50K sözlük, 1024 bağlam.

### 8.4 Korece

- **KURE:** bge-m3 tabanlı; blog yazısına göre hard negative ve **cross-encoder damıtması** kullanıyor ([blog](https://yjoonjang.medium.com/koe5-%EC%B5%9C%EC%B4%88%EC%9D%98-%ED%95%9C%EA%B5%AD%EC%96%B4-%EC%9E%84%EB%B2%A0%EB%94%A9-%EB%AA%A8%EB%8D%B8-multilingual-e5-finetune-22fa7e56d220)) **[blog özeti, makaleyi okumadım]**. `bge-m3-korean` yalnızca KorSTS/KorNLI üzerinde denetimli fine-tune. Çoklu öğretmenli Korece embedding damıtması bulamadım.
- **HanjaBridge:** Korece LLM'lerde Hanja (Çince kökenli) artırımı; farklı tokenizer'lar arasında *token düzeyinde* damıtma tartışması içeriyor, ama embedding değil LLM.

### 8.5 Fince / Macarca ve diğerleri

- Fince veya Macarca için damıtılmış embedding modeli **bulamadım**. Arama sonuçları yalnızca Türkçe örneği (magibu), Reimers & Gurevych çerçevesini ve Bangla (BanglaEmbed, çapraz dilli damıtma, [arXiv 2411.15270](https://arxiv.org/abs/2411.15270)) / Eski Yunanca ([arXiv 2308.13116](https://arxiv.org/abs/2308.13116)) gibi düşük kaynaklı dil örneklerini getirdi; bunların içeriğini okumadım **[DOĞRULANMADI]**.

### 8.6 Sondan eklemeli dillerde çoklu öğretmenli damıtma: durum

| Dil | Çoklu öğretmenli embedding damıtması? | Not |
|---|---|---|
| Japonca | **Evet:** GLuCoSE-v2 (3 öğretmen) | Ayrıntı yok, ablation yok |
| Türkçe | Hayır (yalnızca tek öğretmen) | Bizim boşluğumuz |
| Korece | Bulunamadı | Cross-encoder tek öğretmen (KURE) |
| Fince / Macarca | Bulunamadı | |

---

## 9. Sentez: bizim tasarım kararları için çıkarımlar

1. **Damıtma tek başına yetmeyebilir; iki aşamalı plan standart.** Türkçe'de MoganBert-Embed Faz 1 ile 63,53, Faz 2 + soup ile 68,30'a çıkıyor; EmbeddingGemma, Ruri ve GLuCoSE-v2 hep damıtma + contrastive birleşimi. Projede iki aşama planlanmalı: (1) çoklu öğretmenli damıtma ile embedding uzayını kurmak, (2) isteğe bağlı contrastive/denetimli ince ayar.
2. **Ham ModernBERT-TR anizotropik olacak.** MoganBert'te ham MLM cosine'ı 0,98. Faz 1'de GOR gibi bir düzenleyici ve normalizasyon ele alınmalı. Ham modelin kendi anizotropisini ilk ölçüm olarak raporlamak iyi bir "önce/sonra" göstergesi.
3. **Boyut uyumsuzluğu için üç kanıtlı yol var:** (a) Jasper: öğretmen vektörlerini birleştir, öğrenciyi geniş çıktıya eşle (12.288); (b) Formont: her öğretmen için ayrı çekirdek; (c) MoganBert/Sentence-Transformers: tek öğretmen için öğrenilebilir projeksiyon. Bizim 3 öğretmenli kurulum için (b) ve "öğretmen başına ayrı projeksiyon başlığı + toplam kayıp" doğal ve uygulaması kolay; (a) ise en basit.
4. **Kayıp seçenekleri:** cosine (MoganBert, magibu), ℓ2 (LEAF), benzerlik matrisi MSE + göreli sıralama (Jasper), Gaussian NLL (Formont), skor dağılımına KL (Ruri, MoganColBERT). Bu seçeneklerin **Türkçe'de ve çoklu öğretmen altında karşılaştırılması** başlı başına katkı olur. Jasper'ın `L_sim` ve `L_resim` kayıpları *ilişkisel* olduğu için öğretmen boyutlarından bağımsız; farklı boyutlu öğretmenleri ortak bir kayıpla birleştirmenin en temiz yolu bu olabilir (benim yorumum).
5. **Cross-encoder öğretmen (skor damıtması)** embedding öğretmenlerinden farklı bir sinyal verir ve retrieval'ı iyileştirir (Ruri, MoganColBERT). `bge-reranker-v2-m3` Türkçe'de denenmiş. Retrieval zayıflığını (MoganBert-Embed'in en zayıf noktası) gidermek için embedding öğretmenlerine eklenebilir. Ruri bunun kararsızlık yarattığını belirtmiş; dikkat.
6. **Veri etiket gerektirmiyor:** Jasper 8M metin, Formont ~6M, magibu 580K (!). Hepsi etiketsiz. Önce küçük (≈500K-1M) bir korpusla hızlı deneyler, sonra büyütme planı mantıklı. Jasper veri artırımı (cümle parçalama, kelime karıştırma) kullanmış; Türkçe için ek olarak morfolojik varyasyon (ek çıkarma) denenebilir (benim önerim, denenmemiş).
7. **Girdi biçimi tutarlılığı:** Öğrenci öğretmenle aynı girdi formatını görmeli, öğretmene talimat verilmemeli (MoganBert). Qwen3 ve e5 talimat/önek istiyor, EmbeddingGemma'nın kendi öneki var. Bu, çoklu öğretmenle karmaşıklaşır: her öğretmenin kendi önek şablonu için tutarlı bir kural gerekir.
8. **Tokenizer farkı:** Bizde öğrenci Türkçe WordPiece (50K); öğretmenler XLM-R / Qwen / Gemma tokenizer'ları. Hepsi **cümle düzeyinde** vektör ürettiği için tokenizer uyumsuzluğu sorun değil (magibu ve MoganBert de cümle düzeyinde). Token düzeyli damıtma (MoL, HanjaBridge) gerekmiyor.
9. **Beklenti yönetimi:** Tek öğretmenli Qwen3-8B damıtması öğretmenin %99,5'ine ulaştı (MoganBert); magibu öğretmeni STS'de geçti. Yani çoklu öğretmenle kazanç, *öğretmenin tavanına yakın bir tabanı geçmek* anlamına gelecek. Çoklu öğretmenin değeri muhtemelen (a) görev kırılımlarında (retrieval Qwen3'ten, STS e5'ten) ve (b) öğretmen gürültüsünü ortalamasında; bunu ölçmek için aşağıdaki ablation planı gerekli.
10. **Eksik olan kanıt ve bizim katkımız:** Bulduğum hiçbir çalışmada (Jasper, Formont, GLuCoSE-v2) **"tek öğretmen vs çoklu öğretmen" ablation'ı rapor edilmemiş** (Formont'ta olup olmadığını bilmiyorum; okuduğum kısımda görmedim). Önerilen deney matrisi:
    - Her öğretmen tek başına (3-4 koşu);
    - Öğretmen birleşimleri: ikili, üçlü, dörtlü;
    - Birleştirme yöntemi: concat, ayrı başlıklar + ortalama kayıp, görev-bilinçli ağırlık, öğretmen uyuşmazlığına göre ağırlık;
    - Kayıp türü: cosine vs benzerlik-matrisi vs Gaussian NLL;
    - Aşama: yalnız damıtma vs damıtma + contrastive;
    - Çıktı: TR-MTEB kategori kırılımı, öğretmen tavanına oran, Mizan skoru.
11. **Dikkat edilecek riskler:** Öğretmen yanlılığı öğrenciye geçer (magibu, MoganColBERT'in kendi sınırlılıkları); damıtma kararsız olabilir (Ruri); LLM-üretimi veri doğrulanmazsa gürültü getirir (MoganBert'in duygu üçlüleri, Ruri'nin %20 filtreleme gereği).

---

## 10. Skor tabloları

> **Uyarı:** Her çalışma kendi protokolüyle ölçtü (farklı görev kümeleri, farklı ortalama yöntemleri, farklı dil). Tablolar *aynı çalışma içinde* kıyaslanabilir; çalışmalar arası kıyas kabaca yöndür. Tüm sayılar kaynaklarda yazdığı gibi, yazarların kendi bildirimleridir.

### 10.1 ModernBERT-TR ailesi (öğrencimizin ailesi)

#### 10.1.1 Önemli keşif: laboratuvar zaten bir damıtılmış embedding modeli yayımlamış

`ytu-ce-cosmos` HF koleksiyonunda ([ModernBERT-TR koleksiyonu](https://huggingface.co/collections/ytu-ce-cosmos/modernbert-tr)) hazır embedding modelleri var. **Bunlar bizim projenin doğrudan baseline'ı.**

| Model | Taban | Ne yapıldı | Skor |
|---|---|---|---|
| [`modernbert-tr-embed`](https://huggingface.co/ytu-ce-cosmos/modernbert-tr-embed) | `modernbert-tr-base` (8K bağlam) | **Tek öğretmenli damıtma:** Qwen3-Embedding-8B, ~7,9M Türkçe pasaj, Jasper/Stella tarifi (cosine + Gram matrisi + CoSENT tarzı sıralama kaybı), ardından retrieval / çok görevli / çapraz dilli üç ince ayar ve ağırlık ortalaması | **MTEB(Turkish) ortalaması 68,14** |
| [`modernbert-tr-1k-embed`](https://huggingface.co/ytu-ce-cosmos/modernbert-tr-1k-embed) | Kartta belirtilmemiş | Damıtma **yok**; WeightedCachedMNRL (contrastive), 625.121 örnek, 1 epoch, 2,7 saat, bağlam 512, 768 boyut | **Kartta benchmark yok** |
| `modernbert-tr-reranker` ve diğerleri | | Reranker, ABSA, NER, intent, guardrail | Bu çalışmayla ilgisiz |

**`modernbert-tr-embed` ayrıntıları (model kartından):**

- Öğretmenin 4096 boyutlu vektörleri 768'e **izdüşürülüyor** (kart "projected down" diyor).
- İnce ayar: (1) retrieval: InfoNCE (batch içi + hard negative) + öğretmenin sıralamasına KL terimi; (2) çok görevli: retrieval + NLI + CoSENT-STS + SupCon sınıflandırma + cosine-damıtma tekrarı; (3) çapraz dilli: çok görevli tarif, ağırlıkları yeniden dengelenmiş, İngilizce retrieval pasajları eklenmiş. Sonra ağırlık ortalaması.
- Veri: msmarco-tr, SQuAD-TR, `boun-tabi/nli_tr`, STS-B-TR, Türkçe sınıflandırma alanı metinleri (yorum, haber, sosyal). Yazarlar, MTEB(Turkish) test bölümleriyle sızıntıyı metin hash'iyle kontrol ettiklerini söylüyor.
- Lisans: Apache-2.0.

**Skorlar (MTEB(Turkish), kategori ortalamaları, kartın karşılaştırma tablosu):**

| Model | Param | Retr. | Clf. | Pair Clf. | Kümeleme | STS | Bitext | **Ortalama** |
|---|---|---|---|---|---|---|---|---|
| **ModernBERT-TR-Embed** | 150M | 59,4 | **76,5** | **69,2** | **63,3** | 77,6 | 94,1 | **68,14** |
| turkish-e5-large | 560M | 61,5 | 72,6 | 62,8 | 60,9 | 80,0 | **99,3** | 67,17 |
| Qwen3-Embedding-4B | 4B | **63,1** | 70,2 | 60,1 | 61,3 | 77,0 | 97,8 | 66,69 |
| multilingual-e5-large | 560M | 61,7 | 69,2 | 65,6 | 60,8 | **81,0** | 99,0 | 66,56 |
| harrier-oss-v1-0.6b | 600M | 60,1 | 71,1 | 58,6 | 63,3 | 74,5 | 98,6 | 65,57 |

Tekil görev skorları (nDCG@10, kendi bildirimi, liste eksik çünkü 18 değerlendirme kartta gizli): ArguAna-TR 50,01; CQADupstack-Gaming-TR 56,37; FiQA2018-TR 46,20; MSMARCO-TR 57,87; NFCorpus-TR 9,47; Quora-TR 95,18; SCIDOCS-TR 3,37; MNLI-TR (max AP) 65,92.

> **Bağlantı:** MoganBert makalesinin tablosundaki "ModernBERT-TR, 149M, 8192 bağlam" satırı (genel 68,13; Retr. 59,41; Pair 69,21; STS 77,61; Clf. 76,51; Kümeleme 63,20; Bitext 94,04) bu kartın sayılarıyla neredeyse aynı (68,14; 59,4; 69,2; 77,6; 76,5; 63,3; 94,1). Bu satırın `modernbert-tr-embed` olduğunu **sayıların örtüşmesinden çıkarıyorum**, makalede model adı açıkça yazmıyor. Doğruysa MoganBert-Embed (68,30) bu modelden yalnızca **+0,17** önde.

**Bu bizim proje için ne demek?**

1. **Tek öğretmenli Qwen3-8B damıtması aynı laboratuvarda zaten yapılmış** (68,14). Projede "Qwen3 tek öğretmen" bir *baseline*; yenilik ancak çoklu öğretmenle bu sayının üstüne çıkmak veya öğretmen kombinasyonlarını analiz etmek olabilir.
2. Hedef sayı net: **TR-MTEB/MTEB(Turkish) ≈ 68,1-68,3** üstü anlamlı bir kazanç. Qwen3-8B öğretmenin tavanı 68,66 (MoganBert makalesi), yani tek öğretmen için boşluk çok dar. Çoklu öğretmenin tavanı belirsiz.
3. Bu iki model *farklı taban* kullanıyor (aşağıda): bizim öğrenci `modernbert-tr-base-1k`, lab'ın embed modeli ise `modernbert-tr-base` (8K). Adil kıyas için hangi tabanın kullanılacağını netleştirmek gerekir.

#### 10.1.2 Taban (MLM) modellerin skorları

Bu skorlar *embedding* değil, fine-tune/linear-probe NLU skorları.

| | `modernbert-tr-base-1k` (**bizim öğrenci**) | `modernbert-tr-base` |
|---|---|---|
| Param | ~150M | 149,4M |
| Bağlam | 1.024 | 8.192 |
| Sözlük | 50K WordPiece | 50.008 cased WordPiece |
| Ön eğitim | ~144,4B token (FineWeb-2 TR 41,2B + BertTurk x5 31,0B, 2 epoch), 4xH100, 623 GPU-saat | 1k karışımı (%81,4 Türkçe, %8,5 İngilizce, %10 kod, %0,1 paralel metin): 200B token (1k) + 20B (decay) + ~100B (düşük LR devamı) + 50B (8K genişletme) + 20B (8K decay); `main`, 1k decay soup'u (%50) ile 8K terminal decay'in (%50) ortalaması |
| Kendi bildirdiği | Linear probing (11 görev) **60,2**; TabiBench (28 görev) **77,28** | "Dondurulmuş encoder değerlendirmesinde lider", "TabiBench'te 2. sırada" (**sayısal skor yok**) |

Linear probing (11 Türkçe görev, dondurulmuş encoder, `ModernBERT-TR 1k` kartı):

| Model | Param | Ort. |
|---|---|---|
| **ModernBERT-TR 1k** | 150M | **60,2** |
| mmBERT | 307M | 54,9 |
| Turkish-E5-large | 560M | 53,2 |
| TabiBERT | ~150M | 49,1 |
| BERTurk | 111M | 35,3 |

TabiBench tam fine-tuning (kartta yalnızca ortalamalar, kategori skorları yok):

| Model | Ort. |
|---|---|
| TabiBERT | 77,58 |
| **ModernBERT-TR 1k** | **77,28** |
| BERTurk | 75,96 |

- **[ÇELİŞKİ]** TabiBERT makalesi (v4) TabiBench'te TabiBERT'in ortalamasını **77,28**, BERTurk'ünkini **75,66** veriyor (27 veri seti); model kartı ise TabiBERT için 77,58, BERTurk için 75,96 (28 görev) yazıyor. ModernBERT-TR'nin 77,28'i, makaledeki TabiBERT sayısıyla birebir aynı; bu bir rastlantı ya da karıştırma olabilir. Ayrıca kart ModernBERT-TR'nin 5 kategoride (metin sınıflandırma, STS, NLI, akademik, IR) önde olduğunu söylerken, makale aynı iddiayı TabiBERT için (NLI, QA, akademik, IR, kod) yapıyor. ModernBERT-TR'nin skorlarını **kendi ölçümümüzle** yeniden üretmeden rapora koymayın.
- Makale tablosunun kategori bazlı değerleri (TabiBench, TabiBERT / BERTurk / mmBERT): metin sınıflandırma 83,44 / 83,42 / 82,54; token sınıflandırma 93,42 / 93,67 / 93,81; STS (Pearson) 84,74 / 85,33 / 87,05; NLI 84,51 / 84,33 / 84,38; QA (F1) 69,71 / 60,16 / 71,47; akademik 70,06 / 69,00 / 70,34; IR (nDCG@10) 75,44 / 74,84 / 76,20; kod retrieval 56,95 / 54,54 / 66,02; ortalama 77,28 / 75,66 / 78,98. **ModernBERT-TR bu tabloda yok.**

### 10.2 Türkçe damıtma çalışmalarının skorları

**MoganBert-Embed** (Qwen3-Embedding-8B'den; TR-MTEB 26 görev)

| | Genel | Retr. | Pair Clf. | STS | Clf. | Kümeleme | Bitext |
|---|---|---|---|---|---|---|---|
| MoganBert-Embed (149M) | **68,30** | 59,58 | 69,32 | 79,49 | 75,36 | 66,46 | 97,24 |
| ModernBERT-TR (muhtemelen `modernbert-tr-embed`, bkz. 10.1.1) | 68,13 | 59,41 | 69,21 | 77,61 | 76,51 | 63,20 | 94,04 |
| Qwen3-Embedding-8B (öğretmen) | 68,66 | 64,46 | 62,76 | 80,04 | 72,95 | 63,01 | 98,18 |
| mE5-large-instruct | 67,47 | 61,05 | 63,03 | 81,23 | 73,70 | 61,87 | 98,99 |
| BGE-M3 | 67,36 | 60,26 | 68,72 | 79,60 | 72,79 | 60,80 | 98,99 |
| Mursit-Large-TR | 62,00 | 55,70 | 55,19 | 74,60 | 68,66 | 61,53 | 86,72 |
| trmteb-tr-embed | 59,71 | 52,54 | 56,81 | 74,96 | 70,62 | 63,17 | 37,75 |

Aşama bazlı: yalnızca Faz 1 (damıtma) 63,53; Faz 2 v3 66,97; nihai 68,30 (muhtemelen soup). Sıfır-atış retrieval nDCG@10: 0,2361 → 0,5927 (Faz 1) → 0,6786 (Faz 2 v1). Ham modelin ortalama cosine'ı 0,9841 → 0,0851.
Ek: MoganBert-TR (taban) TrGLUE 78,41, TabiBench 77,73.

**embeddingmagibu-200m** (EmbeddingGemma-300M'den; makalenin kendi TR-MTEB protokolü)

| | Öğrenci | Öğretmen |
|---|---|---|
| Ortalama | 63,9 (26 modelden 7.) | 65,2 |
| Retr. | 72,2 | 75,9 |
| Clf. | 68,5 | 71,8 |
| Kümeleme | 61,4 | 62,4 |
| STS | 77,5 | 72,9 |
| NLI | 67,9 | 60,6 |
| Bitext | 97,0 | 96,8 |
| STSbTR Pearson / Spearman | 77,55 / 77,45 | 73,84 / 72,92 |

Öncül `embeddingmagibu-152m`: STSbTR 72,92 / 71,84. Mizan: MTEB 59,25, hukuk 45,59 (11. sıra). Bu makalenin retrieval değerleri (72,2) MoganBert protokolünden (59-64) çok yüksek: görev/kategori tanımı farklı, kıyaslamayın.

**MoganColBERT-TR** (bge-reranker-v2-m3'ten; 5 Türkçe BEIR seti, ortalama)

| Model | Param | nDCG@10 | nDCG@100 | R@100 | mAP | Genel |
|---|---|---|---|---|---|---|
| mLateOn | 307M | 37,23 | 40,72 | 63,59 | 28,98 | 42,63 |
| **MoganColBERT-TR** | 149M | 31,81 | 35,53 | 56,98 | 25,13 | 37,36 |
| ColmmBERT-base-TR | 307M | 29,21 | 32,54 | 53,19 | 22,29 | 34,31 |
| ColmmBERT-small-TR | 141M | 27,67 | 30,96 | 51,27 | 21,16 | 32,77 |
| LFM2.5-ColBERT-350M | 353M | 20,28 | 23,27 | 41,85 | 14,83 | 25,06 |

Öğretmenin kendi skoru raporlanmamış. NFCorpus-TR'de iki ColmmBERT modelinin gerisinde.

**Damıtmasız Türkçe referanslar** (aynı TR-MTEB; karşılaştırma için): `turkish-e5-large` 67,17 (Retr. 61,5; Clf. 72,6; Pair 62,8; Kümeleme 60,9; STS 80,0; Bitext 99,3). Mizan sıralamasında EmbeddingGemma 65,42, bge-m3 62,87, TurkEmbed4STS 62,42 (Bölüm 1.2).

### 10.3 İngilizce / dilden bağımsız

| Çalışma | Öğretmen sayısı | Öğrenci | Skor |
|---|---|---|---|
| **Jasper** | 2 (NV-Embed-v2, stella 1.5B; v1'de bge-en-icl şüphesi) | 1,9B | MTEB 56 veri seti **71,54** (3. sıra); kategori: Clf. 88,49, Kümeleme 58,04, Pair 88,07, Reranking 60,91, Retr. 61,33, STS 84,67, Özet 31,42. Öğretmenler: NV-Embed-v2 72,31 (7,85B), stella 71,19 (1,54B) |
| **Formont ve ark.** | 4 | Snowflake xs 23M / s 33M / m 109M / l 335M | MTEB 33 görev, sınıflandırma ort.: xs 74,0; s 75,4; m 76,7; l 76,5 (karşılaştırma: GIST-xs 72,7, GIST-s 76,1). Kümeleme/STS tabloları okunamadı, yazarlar ılımlı kazanç diyor |
| **LEAF** | 1 | MiniLM-L6, 23M | `leaf-ir`: BEIR 53,9 (öğretmen arctic-embed-m-v1.5'in %96,1'i), asimetrik 54,8 (%97,7). `leaf-mt`: MTEB v2 İng. 59,4 (%95,8), asimetrik 60,1 (%96,9) |
| **EmbeddingGemma** | 1 (Gemini Embedding) | 300M | MTEB çok dilli v2: **61,15**; ablation: enc-dec 60,4 / dec-only 59,7 / rastgele 45,2; soup 61,2 vs 60,1-60,4 |

### 10.4 Japonca (JMTEB, 16 veri seti)

| Model | Öğretmen / damıtma | Ort. | Retr. | STS | Clf. | Reranking | Kümeleme | Pair |
|---|---|---|---|---|---|---|---|---|
| **GLuCoSE-base-ja-v2** | **3 embedding öğretmeni** (E5-mistral, gte-Qwen2, mE5-large) | 72,23 | 73,36 | 82,96 | 74,21 | 93,01 | 48,65 | 62,37 |
| GLuCoSE-base-ja (v1) | - | 67,29 | 59,02 | 78,71 | 76,82 | 91,90 | 49,78 | 66,39 |
| Ruri-small | cross-encoder (Ruri-Reranker-large) | 71,53 | 69,41 | 82,79 | 76,22 | 93,00 | 51,19 | 62,11 |
| Ruri-base | " | 71,91 | 69,82 | 82,87 | 75,58 | 92,91 | 54,16 | 62,38 |
| Ruri-large | " | 73,31 | 73,02 | 83,13 | 77,43 | 92,99 | 51,82 | 62,29 |
| Ruri v3-310m | kartta damıtma belirtilmemiş | 77,24 | 81,89 | 81,22 | 78,66 | 93,43 | 55,69 | 62,60 |
| multilingual-e5-large | - | 70,90 | 70,98 | 79,70 | 72,89 | 92,96 | 51,24 | 62,15 |
| OpenAI text-embedding-3-large | - | 74,05 | 74,48 | 82,52 | 77,58 | 93,58 | 53,32 | 62,35 |

Ruri ablation: contrastive ön eğitim olmadan retrieval skoru small 56,62 (normalde 69,41), base 52,99 (69,82), large 57,84 (73,02). GLuCoSE-v2'nin v1'e göre kazancı **+4,9 puan** (67,29 → 72,23), ama bu kazancın ne kadarı damıtmadan ne kadarı sonraki contrastive aşamalardan geliyor bilinmiyor. Not: GLuCoSE-v1 ortalaması Ruri makalesinde (mikro ortalama), v2 kartında (makro ortalama) farklı yöntemle verilmiş olabilir; satırlar kesin kıyaslanabilir değil.

### 10.5 Skorlardan çıkan genel desenler

- **Öğrenci öğretmenin tavanına çok yakın:** MoganBert %99,5 (68,30/68,66), magibu %98 (63,9/65,2), LEAF %96.
- **Öğrenci STS ve NLI'da öğretmeni geçebiliyor, retrieval'da geride:** magibu STS 77,5 vs 72,9, NLI 67,9 vs 60,6, retrieval 72,2 vs 75,9; `modernbert-tr-embed` ve MoganBert'te de retrieval en zayıf kategori (59,4-59,6; öğretmen Qwen3-8B 64,5).
- **Tek öğretmenli Qwen3-8B damıtması iki bağımsız çalışmada ~68,1-68,3'e geliyor** (lab'ın embed modeli ve MoganBert). Bu, aynı tarifle başka kazanç beklememek için güçlü bir işaret.
- **Çoklu öğretmenin değeri hiçbir yerde izole edilmemiş:** Jasper, Formont ve GLuCoSE-v2'de tek vs çoklu öğretmen karşılaştırması görmedim.
- **Damıtma + sonraki aşamalar standart:** MoganBert (63,5 → 68,3), `modernbert-tr-embed` (damıtma + 3 ince ayar + ağırlık ortalaması), EmbeddingGemma, Ruri hep birleşik tarif.

---

## 11. Kaynaklar

**Öğrenci model ve yakın çalışmalar**
- [ModernBERT-TR koleksiyonu](https://huggingface.co/collections/ytu-ce-cosmos/modernbert-tr) · [modernbert-tr-embed](https://huggingface.co/ytu-ce-cosmos/modernbert-tr-embed) · [modernbert-tr-1k-embed](https://huggingface.co/ytu-ce-cosmos/modernbert-tr-1k-embed) · [modernbert-tr-base](https://huggingface.co/ytu-ce-cosmos/modernbert-tr-base)
- [ModernBERT-TR-base-1k model kartı](https://huggingface.co/ytu-ce-cosmos/modernbert-tr-base-1k)
- [MoganBert-TR / MoganBert-Embed (arXiv 2608.25768)](https://arxiv.org/abs/2608.25768)
- [Adapting Multilingual Embedding Models to Turkish via Tokenizer Surgery and Offline Distillation (arXiv 2605.29992)](https://arxiv.org/abs/2605.29992)
- [TurkEmbed (arXiv 2511.08376)](https://arxiv.org/abs/2511.08376) · [TurkEmbed4Retrieval (arXiv 2511.07595)](https://arxiv.org/abs/2511.07595)
- [Mecellem Models (arXiv 2601.16018)](https://arxiv.org/abs/2601.16018)
- [TabiBERT ve TabiBench (arXiv 2512.23065)](https://arxiv.org/abs/2512.23065)
- [Jasper and Stella (arXiv 2412.19048)](https://arxiv.org/abs/2412.19048)
- [MoganColBERT-TR (arXiv 2608.26344)](https://arxiv.org/abs/2608.26344)
- [Learning Task-Agnostic Representations through Multi-Teacher Distillation, NeurIPS 2025 (arXiv 2510.18680)](https://arxiv.org/abs/2510.18680)
- [LEAF (arXiv 2509.12539)](https://arxiv.org/abs/2509.12539)
- [EmbeddingGemma (arXiv 2509.20354)](https://arxiv.org/abs/2509.20354)
- [Translation Aligned Sentence Embeddings for Turkish (arXiv 2311.09748)](https://arxiv.org/abs/2311.09748)
- [Reimers & Gurevych, Making Monolingual Sentence Embeddings Multilingual using Knowledge Distillation (arXiv 2004.09813)](https://arxiv.org/abs/2004.09813)
- [Sentence-Transformers model distillation dokümanı](https://www.sbert.net/examples/training/distillation/README.html)
- [Ruri: Japanese General Text Embeddings (arXiv 2409.07737)](https://arxiv.org/abs/2409.07737) · [Ruri v3 kartı](https://huggingface.co/cl-nagoya/ruri-v3-310m)
- [GLuCoSE-base-ja-v2](https://huggingface.co/pkshatech/GLuCoSE-base-ja-v2)
- [KURE blog yazısı](https://yjoonjang.medium.com/koe5-%EC%B5%9C%EC%B4%88%EC%9D%98-%ED%95%9C%EA%B5%AD%EC%96%B4-%EC%9E%84%EB%B2%A0%EB%94%A9-%EB%AA%A8%EB%8D%B8-multilingual-e5-finetune-22fa7e56d220)
- [BanglaEmbed (arXiv 2411.15270)](https://arxiv.org/abs/2411.15270) · [Ancient Greek multilingual KD (arXiv 2308.13116)](https://arxiv.org/abs/2308.13116) (içeriği okunmadı)

**Benchmark'lar**
- [TR-MTEB (EMNLP Findings 2025)](https://aclanthology.org/2025.findings-emnlp.471/) · [trmteb HF org](https://huggingface.co/trmteb)
- [Mizan leaderboard](https://huggingface.co/spaces/newmindai/Mizan)
- [TR-TEB (ResearchGate)](https://www.researchgate.net/publication/407438143_TR-TEB_Turkish_Text_Embedding_Benchmark)
- [TurkColBERT blog](https://huggingface.co/blog/nmmursit/late-interaction-models)
- [TrGLUE / SentiTurca (arXiv 2512.22100)](https://arxiv.org/html/2512.22100v1)
- [MMTEB (arXiv 2502.13595)](https://arxiv.org/abs/2502.13595)

**Öğretmen model kartları**
- [Qwen3-Embedding-8B](https://huggingface.co/Qwen/Qwen3-Embedding-8B)
- [EmbeddingGemma-300M](https://huggingface.co/google/embeddinggemma-300m) · [Gemma Terms of Use](https://ai.google.dev/gemma/terms)
- [BGE-M3](https://huggingface.co/BAAI/bge-m3)
- [multilingual-e5-large-instruct](https://huggingface.co/intfloat/multilingual-e5-large-instruct)
- [gte-multilingual-base](https://huggingface.co/Alibaba-NLP/gte-multilingual-base)
- [llama-embed-nemotron-8b](https://huggingface.co/nvidia/llama-embed-nemotron-8b)
- [jina-embeddings-v3](https://jina.ai/models/jina-embeddings-v3/)
- [snowflake-arctic-embed-l-v2.0](https://huggingface.co/Snowflake/snowflake-arctic-embed-l-v2.0)

**Veri setleri**
- [Cosmos-Turkish-Corpus-v1.0](https://huggingface.co/datasets/ytu-ce-cosmos/Cosmos-Turkish-Corpus-v1.0)
- [boun-tabi/nli_tr](https://huggingface.co/datasets/boun-tabi/nli_tr) · [emrecan/stsb-mt-turkish](https://huggingface.co/datasets/emrecan/stsb-mt-turkish)
- [TrGLUE](https://huggingface.co/datasets/turkish-nlp-suite/TrGLUE) · [parsak/msmarco-tr](https://huggingface.co/datasets/parsak/msmarco-tr)
- [Turkish paraphrase dataset (ICNLSP 2022)](https://aclanthology.org/2022.icnlsp-1.14.pdf)
- [awesome-turkish-language-models](https://github.com/kesimeg/awesome-turkish-language-models)
