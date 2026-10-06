# Destek ve kovaryans takip protokolü

2026-09-05 02:04 +03:00;Codex;gpt-6-astra xhigh;SOFTMCC-COMPLETION-20260905

## Material Passport

Type: code_experiment_plan. Status: prospective_draft_before_scientific_generation.
Run: `2026-09-05_codex_vps_support_covariance`. Change: `MCC-DRIFT-20260905-SUPPORT`.
Öncül: `MCC-DRIFT-20260904-DEPENDENCE`, ACCEPT. Önceki 384 blok ve birincil hüküm değişmez.
Yetki: kullanıcının 5 Eylül deneyleri tamamlama, audit ve gerekli revizyonları uygulama talebi. Ayrı agent veya bağımsız Claude turu iddiası yoktur.

## Yakın literatür ve test edilebilir fark

| Yakın çalışma | Yapılmış olan | Bu protokolün karşılaştırma sınırı |
|---|---|---|
| Gama 2013, 10.1007/s10994-012-5320-9 | Prequential değerlendirmede pencere ve unutma | Unutma mekanizması yeni diye sahiplenilmez. |
| Hidalgo 2019, 10.1111/coin.12208 | Performans tahmininde sliding window/fading karşılaştırması | Eşit ağırlık ESS, aynı ortak dağılım, kontrollü seri bağımlılık ve nonlinear SoftMCC uç noktası. |
| Pozzi 2012 ve erratum, 10.1140/epjb/e2012-20697-x; 10.1140/epjb/e2012-30636-6 | Üstel ağırlıklı korelasyonlar | Yeni ağırlıklı korelasyon iddiası yok; Bernoulli-marjinli SoftMCC fonksiyoneli ve destek kaybı sınanır. |
| Itaya 2025, 10.1002/sim.10303 | Hard MCC için IID delta/Fisher-z ve eşlenmiş CI | SoftMCC, seri bağımlılık ve eşit nominal bellekte sonlu destek; hard-MCC yöntemiymiş gibi yeniden adlandırılmaz. |
| Gomes 2023, 10.1109/ICMLA58977.2023.00335 | Etkin prequential AUC-PR; az pozitif desteğinde oynaklık | Destek sorunu ilk bulgu değildir; tam tanımlı SoftMCC düzeltme/coverage/risk nicelikleri hesaplanır. |
| Tenet 2024, 10.1109/BigData62323.2024.10825670 | Etiket-bağımlılığı enjeksiyonu ve öğrenici benchmark'ı | Ortak dağılım değiştirilmeden retain/refresh zinciri; yeni öğrenici önerilmez. |
| OEUVRE 2026, PMLR 300:4267–4275 | Güncel öğrenicinin kaybını tahmin etme; durağan olmayan deneyler de mevcut | Saklı iki model sorgusu gerektiren farklı estimand; aynı baseline diye sahte uygulama yapılmaz. |
| Newey–West 1986/1987, NBER t0055; Econometrica 55:703–708 | Bartlett ağırlıklarıyla PSD HAC kovaryansı ve asimptotik tutarlılık | Standart HAC kontrol olarak kullanılır; kısa sabit bellek için kapsama garantisi çıkarılmaz. |

Tam metin okuma kanıtları: `MD/01_literature/positioning_research_20260904.md` ve `fulltext_completion_20260905.md`. Newey–West NBER PDF 14/14 yapısal PASS, tamamı yerel OCR ile okundu, Eq.(5)/Theorem 1 ayrıca görüntüden doğrulandı; OCR formülleri koda kopyalanmadı.

Falsifiye edilebilir fark: **Sabit kalibre ortak dağılım ve aynı nominal ESS altında, Bartlett-HAC delta aralıklarının IID delta aralıklarına göre kapsama kazanımı, kısa bellek ve düşük sınıf desteği boyunca hedeflenen %95 kapsamayı sağlamaya yetiyor mu?**

Karar: `worth_testing`, `bounded_extension`. Standart araçları yeni yöntem gibi sunmak yerine kapsamı ölçen değerlendirme katkısı. Başarısızlık yeni bir algoritma uydurularak kurtarılmayacak.

## Veri ve bağımsız tekrarlar

Tamamen sentetik altı-durumlu ortak dağılım, 4 Eylül jeneratörüyle aynı tanım: p ∈ {0.01,0.25,0.90}, y|p ~ Bernoulli(p), prevalans {0.50,0.20,0.05}. Orta skor kütleleri {0.20,0.20,0.10}; kalan iki kütle prevalansı tam sağlayacak şekilde çözülür. Başlangıç durağan dağılımdan; her adımda rho olasılıkla ortak (p,y) durumu korunur, aksi halde taze durum çekilir. Rho ∈ {0,0.3,0.6,0.9}. Her katmanda 16 blok ×64 bağımsız yörünge; toplam 12,288 yörünge, her biri 2500 gözlem. Ana seed 20260905; namespace bileşenleri (prevalans index, rho index, blok, smoke bayrağı). Önceki bilimsel yörüngeler tekrar kullanılmaz. Smoke namespace ayrıdır.

Son konumda w ∈ {20,40,100,200,500} için dolu pencere ve lambda=(w−1)/(w+1) fading aynı yörüngede hesaplanır. Fading 2500-konumluk mevcut prefix ile tam normalize edilir; kesilmez. Koşullar bağımsız deney sayılmaz, karşılaştırmalar yörünge-eşlenmiştir. Zaman birimi gözlem indeksidir; hiçbir öğrenici, insan veya dış veri aktarımı yoktur.

## Estimatör ve aralıklar

z=(p,y,py), theta_hat=sum alpha_i z_i, F=(c−ab)/sqrt[a(1−a)b(1−b)]. Aynı sayısal tanımlılık marjini 1e−12 korunur; tanımsız değer NaN + maskedir, eksik gözlemler sonuçtan gizlenmez. Ham F değiştirilmez.

IID plug-in kovaryans: V_iid=Q Sigma_hat, Q=sum alpha²; Sigma_hat=sum alpha_i z_i z_i'−theta_hat theta_hat'. Ağırlıklı ikinci moment tahmini önceki estimator ile aynıdır; sonlu-örneklem düzeltmesi eklenmez.

HAC: u_i=z_i−theta_hat ve v_i=alpha_i u_i. V_HAC(L)=sum_i v_i v_i' + sum_{h=1}^L (1−h/(L+1)) sum_{i=h+1}^T (v_i v_{i−h}'+v_{i−h}v_i'). L1=max(1,floor(w^(1/3))), L2=max(1,floor(sqrt(w))). Bütün lag ve sonlu-prefix ağırlıkları açık kullanılır. L=0 HC kontrolü ayrıca raporlanır; V_HAC(0) fading için V_iid ile aynı değildir. Bartlett çekirdeğinin PSD özelliği korunur; negatif büyük özdeğer program hatası sayılır, clip ile saklanmaz. Floating variance −1e−12'den büyük küçük residual olursa yalnız sqrt için sıfıra yuvarlanır ve sayılır.

Tanısal exact V=K(rho,alpha) Sigma_population; K=sum_i alpha_i² +2sum_{h>=1}rho^h sum_i alpha_i alpha_{i+h}. Oracle girdileri yalnız diagnostiktir, uygulanabilir monitor değildir.

Aralıklar: F(theta_hat) ± 1.959963984540054 sqrt[g(theta_hat)' V g(theta_hat)], V ∈ {IID, HC0, HAC-L1, HAC-L2, exact}. Aralıklar [-1,1]'e clip edilmez; uzunluk ve out-of-range oranı saklanır. Normal delta aralığı sonlu bellek garantisi değildir. Her hücrede tanımlı-aralık oranı, tanımlı koşullu kapsama, toplam yörünge üzerinden tanımlı-ve-kapsıyor oranı ve ortalama uzunluk birlikte verilir.

Düzeltmeler: raw; IID; HAC-L1; HAC-L2; exact-V plug-in Hessian; population-oracle subtraction. Her B=0.5 tr(H(theta_hat)V) değerinin gate'li sürümü de hesaplanır: min(a,1−a,b,1−b)>=0.025 VE min(b,1−b)/Q>=5 ise düzeltme, aksi halde ham skor. Bu eşikler fit edilmez; evrensel validity testi değildir. Gate'li ve gatesiz sürümler eşlenmiş değerlendirilir, gate kabul oranı bildirilir. Ham skor tanımsızsa bütün sürümler tanımsız kalır. Mutlak bias, RMSE, SD ve [-1,1] dışına çıkma oranı saklanır; RMSE bütün tanımlı yörüngelerde, gate tarafından seçilen altküme ile değiştirilmeden hesaplanır.

## Önceden kilitli birincil test ve negatif sonuç politikası

Tek birincil hücre: prevalans .5, rho .6, w=100, fading. Birincil kontrast: HAC-L1 eksi IID için toplam yörünge üzerinden kapsama oranı (tanımsız=başarısız kapsama). Bağımsız yörüngeler üzerinde 5000 eşlenmiş percentile bootstrap, seed95001. “SUPPORTED_COVERAGE_GAIN” ancak fark >=.05 ve iki-yönlü %95 CI altı >0 ise. Aksi halde “NOT_SUPPORTED”. Bu tek karşılaştırmaya çoklu-test düzeltmesi yoktur; kalan 120 hücre açıklayıcı duyarlılık sonuçlarıdır. Başarı, %95 kalibrasyonun sağlandığı anlamına gelmez; nominal kapsama mesafesi ve Monte Carlo standart hatası ayrı raporlanır. N=1024 iken tek kapsama oranının en kötü MCSE'si .015625; paired farkın MCSE'si veriden hesaplanır. Daha küçük etkiler için güç iddiası yoktur.

Eski bağımlılık deneyinin hükmü değişmez. Primary fail sonrası yeni seed, eşik, bandwidth veya yöntem ile kurtarma koşumu yoktur. Sonuçların hepsi, oracle dahil kötüleşmeler ve tanımsızlıklar korunur. Bulgular yayıma hazırlık değil; entegrasyon+audit girdisidir.

## Doğrulama ve kaynak sınırı

Analitik gradient/Hessian sonlu farklarla; HAC doğrudan yoğun Bartlett matrisiyle; PSD; rho=0 exact covariance limiti; theta/cov histogram ve doğrudan çarpım eşitliği sınanacak. Gerçek VPS interruption/resume smoke, temiz çalışmayla tüm bilimsel dizilerde exact eşitlik ve değişmiş binding reddi zorunlu. Ana çalışmadan önce schema-3 candidate→review→decision→final ve `experiment_freeze.py --phase prelaunch` PASS gerekli. Yerel sonuç terfisi, transfer öncesi uzak receipt + arşiv/member hash + yerel negatif kontrolleri ve ayrı hesaplama doğrulaması sonrasında.

Ortam: aynı Oracle Linux aarch64 Python 3.12.3/NumPy 2.4.6; canlı probe eşleşmesi aranacak. Tek worker, BLAS1, nice10, RAM tavan4GiB, disk tabanı5GiB, çıktı tavan1GiB. Mevcut servisler etkilenmez. Beklenen süre smoke ile ölçülüp RUN_PLAN'a yazılacak; başlangıç tahmini 5–15 dakika, hard timeout 3600s. MTA ek hız gerektirmeyen bu çalışma için kullanılmayacak. Kaynak, script/config, ayrı analysis ve bütün ham endpoint dizileri saklanacak; bilimsel kod ana koşum sırasında değişmeyecek.
