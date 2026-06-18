import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, Circle, FancyArrow


# Grid boyutunu 10x10 olarak genişlettim. Daha büyük bir mağaza ortamı için.
GRID_SIZE = 10
MAX_BATTERY = 100
# Başlangıç şarjını düşük tuttum ki robot bölüm içinde mutlaka şarj istasyonuna uğramak zorunda kalsın.
INITIAL_BATTERY = 50
# 10x10 grid'de daha fazla hücre temizleneceği için maksimum adımı da arttırdım.
MAX_STEPS = 300

# Şarj istasyonunu sol üst köşede sabit tutuyorum.
CHARGE_STATION = (0, 0)
# Mağazada 6 raf var, robot bu hücrelere giremiyor. Daha karmaşık bir yol planlaması gerektiriyor.
OBSTACLES = {
    (2, 2),
    (2, 7),
    (5, 3),
    (5, 6),
    (7, 1),
    (7, 8),
}

# Dinamik kirlilik üretimi parametreleri.
# Her bölümde kirli hücre sayısı bu aralıkta rastgele seçiliyor.
DIRT_MIN_CELLS = 11
DIRT_MAX_CELLS = 14
# Kirlilik seviyesi olasılıkları: az kirli daha sık, çok kirli daha nadir.
DIRT_LEVEL_WEIGHTS = [0.45, 0.35, 0.20]  # level 1, 2, 3 olasılıkları

# Aksiyon idlerini sabit olarak tanımladım, kodun içinde rakam yerine isim yazıyorum.
ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3
ACTION_CLEAN = 4
ACTION_CHARGE = 5

# Hareket aksiyonlarının x ve y eksenindeki etkilerini bir tabloda topladım.
ACTION_DELTAS = {
    ACTION_UP: (0, -1),
    ACTION_DOWN: (0, 1),
    ACTION_LEFT: (-1, 0),
    ACTION_RIGHT: (1, 0),
}

# Render başlığında gösterilecek aksiyon isimleri burada tutuluyor.
ACTION_NAMES = {
    ACTION_UP: "Yukari",
    ACTION_DOWN: "Asagi",
    ACTION_LEFT: "Sol",
    ACTION_RIGHT: "Sag",
    ACTION_CLEAN: "Temizle",
    ACTION_CHARGE: "Sarja Git",
}

# Her hücrenin kirlilik seviyesine göre renk atadım. 4 seviye (0-3) var artık.
CELL_COLORS = {
    0: "#90EE90",   # Temiz - yeşil
    1: "#FFF59D",   # Az kirli - açık sarı
    2: "#FFB74D",   # Orta kirli - turuncu
    3: "#E53935",   # Çok kirli - kırmızı
}
OBSTACLE_COLOR = "#404040"


class StoreCleaningEnv:

    def __init__(self, seed: int = 42, randomize_dirt: bool = True):
        # Seed sabit olunca her çalıştırmada aynı sonuç çıkıyor.
        self.rng = np.random.default_rng(seed)
        # Eğitimde her bölüm farklı kirlilik dağılımı için True, sabit test için False.
        self.randomize_dirt = randomize_dirt

        self.robot_pos = CHARGE_STATION
        self.battery = INITIAL_BATTERY
        self.dirt_map = {}
        self.step_count = 0
        self.last_action = None
        self.total_reward = 0.0
        self.charge_visits = 0

        # Görselleştirme için ek durum bilgileri.
        # Robotun gezdiği yol (iz bırakma efekti).
        self.trail = []
        # Robotun son başarılı hareket yönü (yön oku için).
        self.last_move_direction = None
        # Son aksiyonda başarılı şarj/temizlik olup olmadığı (parıltı efektleri için).
        self.charged_this_step = False
        self.cleaned_this_step = False
        # Son temizlenen hücrenin pozisyonu (parıltı için).
        self.last_cleaned_pos = None

        self.reset()

    def _generate_random_dirt(self):
        # Boş geçerli hücreleri topluyorum (engel ve şarj istasyonu hariç).
        available = [
            (x, y)
            for x in range(GRID_SIZE)
            for y in range(GRID_SIZE)
            if (x, y) != CHARGE_STATION and (x, y) not in OBSTACLES
        ]
        # Rastgele kaç tane kirli hücre olacağını seçiyorum.
        n_cells = int(self.rng.integers(DIRT_MIN_CELLS, DIRT_MAX_CELLS + 1))
        # Rastgele konumlar seçip her birine rastgele kirlilik seviyesi atıyorum.
        chosen_indices = self.rng.choice(len(available), size=n_cells, replace=False)
        dirt_map = {}
        for idx in chosen_indices:
            pos = available[int(idx)]
            level = int(self.rng.choice([1, 2, 3], p=DIRT_LEVEL_WEIGHTS))
            dirt_map[pos] = level
        return dirt_map

    def reset(self):
        # Robot her bölümün başında şarj istasyonundan başlasın istiyorum.
        self.robot_pos = CHARGE_STATION
        # Düşük şarjla başlamak robotu istasyona dönmek zorunda bırakıyor.
        self.battery = INITIAL_BATTERY
        # Kirlilik haritasını her bölümde rastgele üretiyorum (randomize=True ise).
        # Bu sayede ajan sadece tek bir durumu değil, genel problemi öğreniyor.
        if self.randomize_dirt:
            self.dirt_map = self._generate_random_dirt()
        else:
            # randomize=False ise sabit varsayılan bir dağılım üretip onu sakla.
            # İlk reset'te üret, sonraki reset'lerde aynısı kullanılsın.
            if not hasattr(self, "_fixed_dirt_map") or self._fixed_dirt_map is None:
                self._fixed_dirt_map = self._generate_random_dirt()
            self.dirt_map = dict(self._fixed_dirt_map)

        self.step_count = 0
        self.last_action = None
        self.total_reward = 0.0
        self.charge_visits = 0

        # Görselleştirme durumları da resetleniyor.
        self.trail = [self.robot_pos]
        self.last_move_direction = None
        self.charged_this_step = False
        self.cleaned_this_step = False
        self.last_cleaned_pos = None

        return self._get_state()

    def step(self, action: int):
        self.last_action = action
        reward = 0.0
        done = False
        info = {}

        # Her adımda efekt flag'lerini sıfırlıyorum, sadece aksiyon başarılı olursa true olur.
        self.charged_this_step = False
        self.cleaned_this_step = False

        # Aksiyondan önceki en yakın kirli hücre mesafesini kaydediyorum.
        # Bu sayede aksiyondan sonra yaklaşma/uzaklaşma farkı kadar küçük ödül/ceza verebilirim.
        dist_before = self._nearest_dirty_distance()

        # Hareket aksiyonlarında önce yeni konumu hesaplıyorum.
        if action in (ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT):
            dx, dy = ACTION_DELTAS[action]
            new_x = self.robot_pos[0] + dx
            new_y = self.robot_pos[1] + dy

            # Grid dışına çıkma veya engele girme durumlarını kontrol ediyorum.
            if (
                new_x < 0
                or new_x >= GRID_SIZE
                or new_y < 0
                or new_y >= GRID_SIZE
                or (new_x, new_y) in OBSTACLES
            ):
                reward = -5
            else:
                self.robot_pos = (new_x, new_y)
                # Başarılı hareket olduğu için yön oku için son yönü kaydediyorum.
                self.last_move_direction = (dx, dy)
                # Trail'e yeni konumu ekliyorum.
                self.trail.append(self.robot_pos)
                # Her adımın küçük bir maliyeti olsun ki ajan gereksiz dolaşmasın.
                reward = -1
            self.battery -= 1

        # Temizleme aksiyonunda hücrenin kirlilik seviyesine bakıyorum.
        elif action == ACTION_CLEAN:
            dirt_level = self.dirt_map.get(self.robot_pos, 0)
            if dirt_level == 3:
                # Çok kirli hücreye en yüksek ödül.
                reward = 20
                self.dirt_map[self.robot_pos] = 0
                self.cleaned_this_step = True
                self.last_cleaned_pos = self.robot_pos
            elif dirt_level == 2:
                # Orta kirli hücreye orta ödül.
                reward = 10
                self.dirt_map[self.robot_pos] = 0
                self.cleaned_this_step = True
                self.last_cleaned_pos = self.robot_pos
            elif dirt_level == 1:
                # Az kirli hücreye küçük ödül.
                reward = 5
                self.dirt_map[self.robot_pos] = 0
                self.cleaned_this_step = True
                self.last_cleaned_pos = self.robot_pos
            else:
                # Zaten temiz olan hücreyi temizlemeye çalışmak israf, küçük ceza veriyorum.
                reward = -2
            self.battery -= 2

        # Şarja gitme aksiyonunda istasyon kontrolü yapıyorum.
        elif action == ACTION_CHARGE:
            if self.robot_pos == CHARGE_STATION:
                # Pil belirli bir seviyenin altındaysa şarj mantıklı.
                # Eşiği 40 yaptım, böylece düşük/kritik banttayken şarj olduğunda ödül alıyor.
                if self.battery <= 40:
                    self.battery = MAX_BATTERY
                    reward = 15
                    self.charge_visits += 1
                    self.charged_this_step = True
                else:
                    # İstasyondayım ama pilim doluysa şarj olmak gereksiz.
                    reward = -2
            else:
                # İstasyon dışında şarja git demek anlamsız olduğu için ceza koyuyorum.
                reward = -3
            self.battery -= 1

        # Pil eksiyse 0'a sabitliyorum, negatif göstermek mantıklı olmazdı.
        if self.battery < 0:
            self.battery = 0

        # Potential-based reward shaping: en yakın kirli hücreye yaklaştıkça küçük bonus.
        # Sadece hareket aksiyonlarında uyguluyorum, temizleme ve şarj aksiyonları
        # zaten kendi ödüllerini almıştı.
        if action in (ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT):
            dist_after = self._nearest_dirty_distance()
            if dist_before is not None and dist_after is not None:
                # Yaklaşmak +0.5, uzaklaşmak -0.5. Bu kararsız oscillation'ı bastırıyor.
                reward += 0.5 * (dist_before - dist_after)

        # Bölüm bitiş kontrolleri burada yapılıyor.
        # Tüm kirli hücreler temizlendiyse bölüm başarıyla bitiyor.
        if all(level == 0 for level in self.dirt_map.values()):
            reward += 100
            done = True
            info["result"] = "success"
        # Pil bittiyse ve robot istasyonda değilse bölüm başarısızla bitiyor.
        elif self.battery <= 0 and self.robot_pos != CHARGE_STATION:
            reward = -100
            done = True
            info["result"] = "battery_dead"
        # Maksimum adım sayısına ulaşıldıysa bölüm timeout ile bitiyor.
        elif self.step_count + 1 >= MAX_STEPS:
            done = True
            info["result"] = "timeout"

        self.step_count += 1
        self.total_reward += reward
        info["charge_visits"] = self.charge_visits
        return self._get_state(), reward, done, info

    def render(self):
        # Her render çağrısında yeni bir figure açıyorum, sonunda kapatmam gerekiyor.
        fig, ax = plt.subplots(figsize=(9, 9.5))

        action_name = ACTION_NAMES.get(self.last_action, "—")
        title = (
            f"Adim: {self.step_count}  |  Aksiyon: {action_name}  |  "
            f"Sarj: %{self.battery}  |  Sarj Ziyareti: {self.charge_visits}  |  "
            f"Toplam Odul: {self.total_reward:.0f}"
        )
        ax.set_title(title, fontsize=10, pad=10)

        # Grid hücrelerini tek tek dolaşarak çiziyorum.
        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                if (x, y) in OBSTACLES:
                    color = OBSTACLE_COLOR
                else:
                    dirt_level = self.dirt_map.get((x, y), 0)
                    color = CELL_COLORS[dirt_level]

                rect = Rectangle(
                    (x - 0.5, y - 0.5),
                    1.0,
                    1.0,
                    facecolor=color,
                    edgecolor="black",
                    linewidth=1.0,
                )
                ax.add_patch(rect)

                # Engellerin üzerine RAF yazısı koyuyorum, anlaşılır olsun diye.
                if (x, y) in OBSTACLES:
                    ax.text(
                        x, y, "RAF",
                        ha="center", va="center",
                        color="white", fontsize=8, fontweight="bold",
                    )

        # İZ BIRAKMA: Robotun gezdiği yolu hafif gri tonlarda gösteriyorum.
        # Eski ziyaretler daha şeffaf, yeni ziyaretler daha koyu görünsün diye alpha
        # yaşıyla orantılı arttırıyorum.
        if len(self.trail) > 1:
            trail_len = len(self.trail)
            for i, pos in enumerate(self.trail[:-1]):
                # Yaş (0 = en eski, 1 = en yeni)
                age_ratio = i / max(trail_len - 1, 1)
                alpha = 0.12 + 0.30 * age_ratio
                ax.scatter(
                    pos[0], pos[1],
                    s=80,
                    c="#1565C0",
                    alpha=alpha,
                    edgecolors="none",
                    zorder=2,
                )

        # Şarj istasyonunu sarı yıldız olarak işaretliyorum.
        cs_x, cs_y = CHARGE_STATION
        ax.scatter(
            cs_x, cs_y,
            marker="*", s=500,
            c="gold", edgecolors="black", linewidths=1.5,
            zorder=3,
        )

        # ŞARJ PARILTISI: Robot bu adımda başarıyla şarj olduysa istasyon çevresinde
        # halka halka genişleyen sarı bir parıltı çiziyorum.
        if self.charged_this_step:
            for radius, alpha in [(0.45, 0.55), (0.7, 0.35), (0.95, 0.18)]:
                glow = Circle(
                    (cs_x, cs_y), radius,
                    facecolor="gold", edgecolor="none",
                    alpha=alpha, zorder=3,
                )
                ax.add_patch(glow)

        rx, ry = self.robot_pos

        # TEMİZLİK PARILTISI: Robot bu adımda başarıyla temizleme yaptıysa
        # küçük yıldızlar çiziyorum, etrafta parlama efekti gibi.
        if self.cleaned_this_step and self.last_cleaned_pos is not None:
            cx, cy = self.last_cleaned_pos
            # Yıldızları sekiz farklı yönde küçük noktalar olarak çiziyorum.
            sparkle_offsets = [
                (0, -0.32), (0, 0.32), (-0.32, 0), (0.32, 0),
                (-0.22, -0.22), (0.22, -0.22), (-0.22, 0.22), (0.22, 0.22),
            ]
            for ox, oy in sparkle_offsets:
                ax.scatter(
                    cx + ox, cy + oy,
                    marker="*", s=100,
                    c="#FFEB3B", edgecolors="#F57F17", linewidths=0.7,
                    zorder=6,
                )

        # Robotu mavi daire olarak çiziyorum, üzerine R harfi koyuyorum.
        ax.scatter(
            rx, ry,
            marker="o", s=500,
            c="royalblue", edgecolors="black", linewidths=1.5,
            zorder=4,
        )
        ax.text(
            rx, ry, "R",
            ha="center", va="center",
            color="white", fontsize=12, fontweight="bold",
            zorder=5,
        )

        # YÖN OKU: Robotun son hareket yönünü gösteren küçük bir ok çiziyorum.
        # Hareket etmediyse yön çizmiyorum.
        if self.last_move_direction is not None:
            dx, dy = self.last_move_direction
            if dx != 0 or dy != 0:
                arrow = FancyArrow(
                    rx, ry, dx * 0.32, dy * 0.32,
                    width=0.06, head_width=0.18, head_length=0.13,
                    color="white", ec="black", linewidth=0.6,
                    length_includes_head=True,
                    zorder=6,
                )
                ax.add_patch(arrow)

        ax.set_xlim(-0.5, GRID_SIZE - 0.5)
        ax.set_ylim(-0.5, GRID_SIZE - 0.5)
        ax.set_xticks(range(GRID_SIZE))
        ax.set_yticks(range(GRID_SIZE))
        ax.set_aspect("equal")
        # Y eksenini ters çeviriyorum ki standart ekran kordinatı gibi gözüksün.
        ax.invert_yaxis()
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

        # Pil seviyesini bir bar olarak alta ekliyorum.
        bar_battery = self.battery / MAX_BATTERY
        # Pil düştükçe bar rengi yeşilden kırmızıya kayıyor.
        if self.battery <= 15:
            bar_color = "#D32F2F"
        elif self.battery <= 40:
            bar_color = "#F57C00"
        elif self.battery <= 75:
            bar_color = "#FBC02D"
        else:
            bar_color = "#388E3C"

        ax.barh(
            -1.5, bar_battery * (GRID_SIZE - 1),
            height=0.35, left=-0.5,
            color=bar_color, edgecolor="black",
        )
        ax.text(
            (GRID_SIZE - 1) / 2 - 0.5, -1.95,
            f"Sarj: %{self.battery}",
            ha="center", va="center", fontsize=10,
        )

        # Lejant ekledim ki izleyen kişi renklerin ne anlama geldiğini bilsin.
        legend_y = GRID_SIZE - 0.2
        legend_items = [
            ("#90EE90", "Temiz"),
            ("#FFF59D", "Az Kirli"),
            ("#FFB74D", "Orta Kirli"),
            ("#E53935", "Cok Kirli"),
            ("#404040", "Engel"),
        ]
        for i, (col, label) in enumerate(legend_items):
            ax.add_patch(
                Rectangle(
                    (-0.4 + i * 2.0, legend_y),
                    0.4, 0.3,
                    facecolor=col, edgecolor="black",
                )
            )
            ax.text(
                0.05 + i * 2.0, legend_y + 0.15,
                label, fontsize=8, va="center",
            )

        ax.set_ylim(GRID_SIZE + 0.5, -2.3)

        # Figure'ı numpy array haline çeviriyorum, GIF üretirken bu şekilde lazım oluyor.
        fig.tight_layout()
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        rgb = rgba[..., :3].copy()

        # Burayı unutursam bellek dolup kod çöküyor, mutlaka kapatmam gerekiyor.
        plt.close(fig)
        return rgb

    def _get_state(self):
        # State olarak (x, y, şarj bandı, mevcut hücre kirliliği, en yakın kirli yönü) döneriyorum.
        # 10x10 grid'de kirli hücre yerleri her bölümde değişebilir, bu yüzden robotun
        # bunların yerini bilmesi şart. En yakın kirli hücrenin yönü pusula görevi görüyor.
        dx_dir, dy_dir = self._nearest_dirty_direction()
        return (
            self.robot_pos[0],
            self.robot_pos[1],
            self._battery_band(self.battery),
            self.dirt_map.get(self.robot_pos, 0),
            dx_dir,
            dy_dir,
        )

    def _nearest_dirty_distance(self):
        # En yakın kirli hücreye olan Manhattan mesafesi.
        # Hiç kirli hücre kalmadıysa None döner.
        dirty_cells = [pos for pos, level in self.dirt_map.items() if level > 0]
        if not dirty_cells:
            return None
        rx, ry = self.robot_pos
        return min(abs(p[0] - rx) + abs(p[1] - ry) for p in dirty_cells)

    def _nearest_dirty_direction(self):
        # En yakın kirli hücreyi Manhattan mesafesi ile buluyorum.
        # Bulamazsam (hepsi temiz) yön (0, 0) dönüyor. İndeks için 0/1/2'ye kaydırıyorum.
        dirty_cells = [(pos, level) for pos, level in self.dirt_map.items() if level > 0]
        if not dirty_cells:
            return (1, 1)

        rx, ry = self.robot_pos
        # Tie-break sırası: yakın mesafe > yüksek kirlilik > küçük x > küçük y.
        # Bu deterministik sıralama oscillation'ı azaltıyor, eşit uzaklıkta her seferinde
        # aynı hücre seçildiği için robot kararsız kalmıyor.
        nearest_pos, _ = min(
            dirty_cells,
            key=lambda p: (
                abs(p[0][0] - rx) + abs(p[0][1] - ry),
                -p[1],
                p[0][0],
                p[0][1],
            ),
        )

        dx = nearest_pos[0] - rx
        dy = nearest_pos[1] - ry
        dx_sign = 0 if dx == 0 else (1 if dx > 0 else -1)
        dy_sign = 0 if dy == 0 else (1 if dy > 0 else -1)

        # -1, 0, 1 → 0, 1, 2 olarak indeks için kaydırıyorum.
        return (dx_sign + 1, dy_sign + 1)

    @staticmethod
    def _battery_band(battery: int) -> int:
        # Sürekli pil değerini 4 banda indirgiyorum çünkü Q-Table'a 100 ayrı değer sığmıyor.
        if battery <= 15:
            return 0
        if battery <= 40:
            return 1
        if battery <= 75:
            return 2
        return 3
