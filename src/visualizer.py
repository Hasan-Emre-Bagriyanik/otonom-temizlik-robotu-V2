import matplotlib

# Agg backend kullanıyorum, ekran olmayan ortamlarda da grafik çizebilsin diye.
matplotlib.use("Agg")

# imageio v2 kullanıyorum çünkü v3 sürümünde uyarı veriyor.
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np


def moving_average(values, window: int = 50):
    # Ham veri çok dalgalı olunca trendi görmek zorlaşıyor.
    # 50-bölüm hareketli ortalama ile eğriyi yumuşatıyorum.
    values = np.asarray(values, dtype=float)
    if len(values) < window:
        return values
    return np.convolve(values, np.ones(window) / window, mode="valid")


def plot_training_rewards(rewards, save_path: str, window: int = 50):
    fig, ax = plt.subplots(figsize=(11, 5))

    # Ham veriyi arka planda yarı saydam çiziyorum.
    ax.plot(rewards, color="lightgray", alpha=0.6, label="Bolum Odulu")

    # Üzerine hareketli ortalama eğrisini ekliyorum, asıl trendi gösteren bu çizgi.
    if len(rewards) >= window:
        ma = moving_average(rewards, window)
        x = np.arange(window - 1, len(rewards))
        ax.plot(x, ma, color="steelblue", linewidth=2, label=f"{window}-Bolum Ortalama")

    ax.set_xlabel("Bolum")
    ax.set_ylabel("Toplam Odul")
    ax.set_title("Egitim Boyunca Odul Grafigi")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def plot_episode_lengths(lengths, save_path: str, window: int = 50):
    # Eğitim ilerledikçe ajan daha verimli olmalı, bu yüzden eğri aşağıya doğru gitmeli.
    fig, ax = plt.subplots(figsize=(11, 5))

    ax.plot(lengths, color="lightgray", alpha=0.6, label="Bolum Uzunlugu")

    if len(lengths) >= window:
        ma = moving_average(lengths, window)
        x = np.arange(window - 1, len(lengths))
        ax.plot(x, ma, color="darkorange", linewidth=2, label=f"{window}-Bolum Ortalama")

    ax.set_xlabel("Bolum")
    ax.set_ylabel("Adim Sayisi")
    ax.set_title("Bolum Uzunlugu Grafigi (Adim Sayisi)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def plot_success_rate(success, save_path: str, window: int = 50):
    # Success listesi 0 ve 1'lerden oluşuyor, ortalamasını alınca başarı oranı çıkıyor.
    fig, ax = plt.subplots(figsize=(11, 5))

    success_arr = np.asarray(success, dtype=float)

    if len(success_arr) >= window:
        ma = moving_average(success_arr, window)
        x = np.arange(window - 1, len(success_arr))
        # Yüzde olarak göstermek için 100 ile çarpıyorum.
        ax.plot(
            x, ma * 100,
            color="seagreen", linewidth=2,
            label=f"{window}-Bolum Basari Orani",
        )

    ax.set_xlabel("Bolum")
    ax.set_ylabel("Basari Orani (%)")
    ax.set_title("Egitim Basari Orani Grafigi")
    # Limitleri biraz geniş tutuyorum ki 0 ve 100 çizgileri kenara yapışmasın.
    ax.set_ylim(-5, 105)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def plot_comparison(q_rewards, random_rewards, save_path: str):
    # Yan yana iki grafik için subplot oluşturuyorum.
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Sol tarafta çizgi grafiği ile karşılaştırma yapıyorum.
    axes[0].plot(q_rewards, color="steelblue", linewidth=2, label="Q-Learning")
    axes[0].plot(random_rewards, color="crimson", alpha=0.6, label="Random Ajan")
    axes[0].set_xlabel("Bolum")
    axes[0].set_ylabel("Toplam Odul")
    axes[0].set_title("Bolum Bazinda Odul Karsilastirmasi")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Sağ tarafta histogram ile ödül dağılımını gösteriyorum.
    axes[1].hist(q_rewards, bins=20, color="steelblue", alpha=0.75, label="Q-Learning")
    axes[1].hist(
        random_rewards, bins=20, color="crimson", alpha=0.55, label="Random Ajan"
    )
    axes[1].set_xlabel("Toplam Odul")
    axes[1].set_ylabel("Bolum Sayisi")
    axes[1].set_title("Odul Dagilimi")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def record_best_episode_gif(env, agent, save_path: str, fps: int = 5, max_steps: int = 300, eps: float = 0.0, n_candidates: int = 50):
    # En iyi bölümü seçmek için n_candidates kadar deneme yapıyorum.
    # Başarılı ve en yüksek ödüllü olanı kaydediyorum.
    saved_epsilon = getattr(agent, "epsilon", None)
    if saved_epsilon is not None:
        agent.epsilon = eps

    best_frames = None
    best_info = None
    best_reward = -1e9
    best_steps = 0

    for attempt in range(n_candidates):
        frames = []
        state = env.reset()
        frames.append(env.render())

        done = False
        steps = 0
        info = {}

        while not done and steps < max_steps:
            action = agent.choose_action(state, training=True)
            next_state, reward, done, info = env.step(action)
            frames.append(env.render())
            state = next_state
            steps += 1

        # Sadece başarılı bölümleri değerlendiriyorum.
        if info.get("result") == "success":
            if env.total_reward > best_reward:
                best_frames = frames
                best_info = info
                best_reward = env.total_reward
                best_steps = steps

    duration = 1.0 / fps
    if best_frames is None:
        raise RuntimeError(f"{n_candidates} denemede basarili bolum bulunamadi.")
    imageio.mimsave(save_path, best_frames, duration=duration, loop=0)

    if saved_epsilon is not None:
        agent.epsilon = saved_epsilon

    print(
        f"GIF kaydedildi: {save_path} ({len(best_frames)} frame, "
        f"adim={best_steps}, odul={best_reward:.0f})"
    )
    return {"steps": best_steps, "reward": best_reward}


def record_episode_gif(env, agent, save_path: str, fps: int = 5, max_steps: int = 300, eps: float = 0.0, max_attempts: int = 30):
    # Eğitilmiş ajanı bir bölüm boyunca çalıştırıp her adımı frame olarak topluyorum.
    # Başarısız bir bölüm yakalarsam max_attempts kez tekrar deniyorum, başarılı olanı kaydediyorum.

    # Env'in rng'sini sistem zamanı ile yeniden seedliyorum ki her çalıştırmada
    # farklı bir kirlilik dağılımı GIF'e düşsün. Aksi halde env hep aynı seed=42 ile
    # başlayıp aynı dağılımı veriyordu.
    import time
    env.rng = np.random.default_rng(int(time.time() * 1e6) % (2**32))

    # Geçici olarak ajanın epsilon'unu ayarlıyorum.
    saved_epsilon = getattr(agent, "epsilon", None)
    if saved_epsilon is not None:
        agent.epsilon = eps

    best_frames = None
    best_info = None
    best_reward = -1e9
    best_steps = 0

    for attempt in range(max_attempts):
        frames = []
        state = env.reset()
        frames.append(env.render())

        done = False
        steps = 0
        info = {}

        while not done and steps < max_steps:
            action = agent.choose_action(state, training=True)
            next_state, reward, done, info = env.step(action)
            frames.append(env.render())
            state = next_state
            steps += 1

        # Başarılı bir bölüm bulduysam onu kullanıyorum ve döngüden çıkıyorum.
        if info.get("result") == "success":
            best_frames = frames
            best_info = info
            best_reward = env.total_reward
            best_steps = steps
            print(f"  GIF: Basarili bolum bulundu ({attempt+1}. denemede, {steps} adim).")
            break

        # Başarısızsa en iyi (en yüksek ödüllü) bölümü saklıyorum.
        if env.total_reward > best_reward:
            best_frames = frames
            best_info = info
            best_reward = env.total_reward
            best_steps = steps

    duration = 1.0 / fps
    imageio.mimsave(save_path, best_frames, duration=duration, loop=0)

    # Orijinal epsilon'u geri yüklüyorum.
    if saved_epsilon is not None:
        agent.epsilon = saved_epsilon

    print(
        f"GIF kaydedildi: {save_path} ({len(best_frames)} frame, {fps} fps, "
        f"sonuc: {best_info.get('result', 'unknown') if best_info else 'unknown'}, "
        f"toplam odul: {best_reward:.0f})"
    )
