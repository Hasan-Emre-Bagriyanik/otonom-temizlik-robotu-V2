# Otonom Magaza Temizlik Robotu

Q-Learning ile bir temizlik robotunun davranisini ogrettigim kucuk bir simulasyon projesi. Reinforcement learning dersi icin yaptim.

Senaryo soyle: gece kapanan bir magazada robot zemini temizleyecek. Sarji bitmeden, raflara carpmadan, cok kirli yerlere oncelik vererek. Bunu kural yazarak cozmek yerine ajana kendi ogrenmesini birakiyoruz.

## Calistirma

```bash
pip install -r requirements.txt
python main.py
```

Egitim normal bir laptopta 1-2 dakika suruyor. Bittiginde `outputs/` klasoru altina grafikler, GIF ve egitilmis Q-Table dusuyor.

## Ortam

5x5'lik bir grid. Her hucrenin kirlilik seviyesi var (temiz / kirli / cok kirli). Bir sarj istasyonu, iki tane raf (engel) sabit konumda.

```text
Sutun:    0     1     2     3     4
       +-----+-----+-----+-----+-----+
  0    |  *  |     |     |  K  |     |
       +-----+-----+-----+-----+-----+
  1    |     |     |  R  |     |  K  |
       +-----+-----+-----+-----+-----+
  2    |     |     | KK  |     |     |
       +-----+-----+-----+-----+-----+
  3    |     |  R  |     |  K  |     |
       +-----+-----+-----+-----+-----+
  4    |  K  |     |     |     | KK  |
       +-----+-----+-----+-----+-----+

  *   : Sarj istasyonu
  R   : Raf (engel)
  K   : Kirli hucre
  KK  : Cok kirli hucre
```

Toplamda 6 tane temizlenecek hucre var (4 kirli + 2 cok kirli). Robotun 6 aksiyonu var: 4 yon hareket, temizle, sarja git.

"Sarja Git" aksiyonu robotu istasyona isinlamiyor. Sadece istasyondaysa pili dolduruyor. Bunu ozellikle boyle yaptim cunku robotun "ne zaman sarja donmem lazim" sorusunu kendisinin ogrenmesini istedim.

## State tasarimi

Ilk basta tum grid'in kirlilik haritasini state'e koymayi dusunmustum ama hesabi yapinca vazgectim:

```text
25 hucre x 3 farkli kirlilik = 3^25 = ~8.5 x 10^11 farkli durum
```

Tablolu Q-Learning icin imkansiz. O yuzden state'i kucult-tum:

```python
state = (robot_x, robot_y, sarj_bandi, mevcut_hucre_kirliligi)
```

Sarji da 0-100 arasi tutmak yerine 4 banda ayirdim (Kritik / Dusuk / Orta / Tam). Boylece toplam 5 * 5 * 4 * 3 = 300 state cikti, Q-Table 300 x 6 = 1800 hucre. Bu boyutla ogrenme makul surede yakinsiyor.

Tabii bu state'in bir dezavantaji var: robot hangi hucrelerin hala kirli oldugunu "gormuyor", sadece bulundugu hucrenin kirliligini biliyor. Bu yuzden robot biraz "kor" geziyor ve tum kirli hucreleri bulana kadar bazen gereksiz dolasiyor. Daha iyi bir state tasarimi yapilabilir ama tabular Q-Learning icin bu yeterli oldu.

## Reward fonksiyonu

Reward'lari ayarlamak en cok ugrastigim kisim oldu. Birkac kez deneme yapip son hali soyle:

| Olay | Odul |
| --- | --- |
| Cok kirli hucreyi temizleme | +20 |
| Kirli hucreyi temizleme | +10 |
| Temiz hucreye temizle aksiyonu | -2 |
| Her adim (bos hareket) | -1 |
| Duvar / engel carpma | -5 |
| Sarjda dusuk pille sarja gitme | +5 |
| Sarjda dolu pille sarja gitme | -2 |
| Sarj sifirlanirsa | -100 |
| Tum kirli hucreler temiz (terminal) | +100 |

Iki nokta onemliydi:

- **Cok kirli ile kirli arasinda fark** koymadigim ilk denemede ajan kirli yerleri rastgele temizliyordu. +10 / +20 farki onceligi netlestirdi.
- **Her adim -1** olmasaydi ajan bos bos dolanmaktan geri durmuyordu. Bu sayede en kisa yolu bulmaya zorlandi.

Sarj sifirlanmasina -100 verdim cunku bu gercek hayatta robotun ortasinda yerde kalmasi demek, agir bir hata.

## Q-Learning

Klasik Bellman guncellemesi:

```python
Q(s,a) = Q(s,a) + alpha * (r + gamma * max(Q(s',a')) - Q(s,a))
```

Hyperparameter'lar:

- alpha = 0.1 (learning rate)
- gamma = 0.95 (discount factor)
- epsilon: 1.0'dan baslayip her bolumde 0.995 ile carparak 0.01'e kadar dusuyor
- 3000 episode, bolum basi maksimum 200 adim
- random seed 42

Alpha'yi once 0.3 yapmistim, Q degerleri cok osilasyon yapiyordu, 0.1'e dusurunce stabil hale geldi. Episode sayisini 3000 sectim cunku 1500 civari ogrenme platosuna ulasiyor, sonrasi sadece dogrulama.

## Sonuclar

3000 bolum sonunda sayisal olarak iyi sonuclar geldi.

### Odul grafigi

Ilk 300 bolum negatif, 500 civari hizla yukselis, 1000'den sonra plato.

![Egitim odul grafigi](outputs/plots/training_rewards.png)

Ilk 100 bolumun ortalama odulu yaklasik -217 idi, son 100 bolumde +138 oldu. Yani ajan basta her bolumde belasini buluyordu, sonunda her bolumden artida cikiyor.

### Basari orani

![Basari orani](outputs/plots/success_rate.png)

Ilk 100 bolumde sadece %2 basari (yani 100 denemenin 2'sinde tum kirli hucreleri temizleyebilmis). 600. bolumden itibaren %90 ustune cikiyor, son 100 bolum ortalamasi %95.

### Bolum uzunlugu

![Bolum uzunlugu](outputs/plots/episode_lengths.png)

Bolum basina ortalama adim sayisi ~95'ten ~43'e dustu. Yani ajan gorevi yapmayi degil, "verimli yapmayi" da ogrendi.

### Q-Learning vs Random

Egitim bittikten sonra epsilon=0 ile (yani sadece ogrenilen politika ile) 100 bolum kosturdum. Ayni ortamda rastgele aksiyon secen bir random ajani da 100 bolum kosturup karsilastirdim:

![Q-Learning vs Random](outputs/plots/q_vs_random.png)

| | Q-Learning | Random |
| --- | --- | --- |
| Ortalama odul | +156 | -280 |
| Ortalama adim | 30 | 104 |
| Basari orani | %100 | %0 |

Random ajan 100 bolumde bir kez bile gorevi tamamlayamadi. Bu da reward sinyallerinin ne kadar etkili oldugunu gosteriyor sanirim.

## Gorsellestirme

Egitim bittikten sonra Q-Table'i yukleyip ajani bir kez epsilon=0 ile kosturdum, her adimi GIF frame'ine cevirdim:

![Final episode](outputs/gifs/final_episode.gif)

GIF'te robotun anlik konumu (mavi daire), hucrelerin kirlilik seviyeleri (yesil/sari/kirmizi), raflar (gri), sarj istasyonu (sari yildiz), sarj seviyesi bari ve secilen aksiyon goruluyor. Bu bolumde robot 30 adimda gorevi bitirdi, +156 odul aldi.

## Gozlemler

Egitim sonunda ajanin sunlari ogrendigini soyleyebilirim:

- Cok kirli hucrelere onceligi kapmasi guzel oldu, +20 reward farki ise yaramis
- Engellere carpma davranisi 200. bolumden sonra cok azaldi
- Verimli yol: 95 adimdan 30 adima dusus
- Sarji belli bir esige dusunce istasyona donme egilimi gozlemleniyor
- Temiz hucreyi tekrar temizlemeye calismak gibi gereksiz aksiyonlar da zamanla yok oldu

Beklemedigim bir sey: ajan bazi bolumlerde sarj seviyesini kullanmadan da tamamlayabilirken yine de istasyona ugruyor. Sanirim sarj_bandi state'e dahil oldugu icin "sarji tam tutmak" ogrenilen bir aliskanlik haline gelmis.

## Sinirlar

5x5 cok kucuk bir grid, kirlilik dagilimi sabit, tek robot. Daha buyuk bir magazada bu yaklasim direkt calismaz - state space patlar. Function approximation (DQN) gerekir. Ama tabular Q-Learning'in temellerini gosterme amaciyla yeterli oldu.

## Kullanilan teknolojiler

Python 3.9+, NumPy, Matplotlib, ImageIO. Hepsi `requirements.txt` icinde.
