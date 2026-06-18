import os

from src.agent import QLearningAgent
from src.environment import StoreCleaningEnv
from src.trainer import evaluate, print_summary, run_random_agent, train
from src.visualizer import (
    plot_comparison,
    plot_episode_lengths,
    plot_success_rate,
    plot_training_rewards,
    record_episode_gif,
)


def create_output_dirs():
    # exist_ok=True parametresi sayesinde klasör zaten varsa hata vermiyor.
    os.makedirs("outputs/plots", exist_ok=True)
    os.makedirs("outputs/gifs", exist_ok=True)


def main():
    print("Otonom Magaza Temizlik Robotu - Q-Learning")
    print("=" * 70)

    create_output_dirs()

    # Aynı seed'i hem ortam hem ajan için kullanıyorum, sonuçların tekrar üretilebilir olması için.
    env = StoreCleaningEnv(seed=42)

    agent = QLearningAgent(
        alpha=0.1,
        gamma=0.95,
        epsilon=1.0,
        epsilon_min=0.01,
        # Büyük state space'te daha fazla keşif gerek, decay'i yavaşlattım.
        epsilon_decay=0.9997,
        # Alpha'yı da yavaşça azaltıyorum ki geç eğitimde Q değerleri stabil otursun.
        alpha_min=0.02,
        alpha_decay=0.9999,
        seed=42,
    )

    # Eğitim aşaması başlıyor.
    # Dinamik kirlilik nedeniyle her bölüm farklı, ajan genelleştirmeyi öğrenmesi için
    # 25000 bölüme çıkardım.
    print("\n[1/5] Egitim baslatiliyor (25000 bolum, dinamik kirlilik)...\n")
    history = train(env, agent, episodes=25000, log_every=1000)

    # Eğitilmiş Q-Table'ı diske kaydediyorum, sonra tekrar eğitmek istemiyorsam yükleyebilirim.
    print("\n[2/5] Q-Table kaydediliyor...")
    agent.save("outputs/q_table.npy")
    print("Kaydedildi: outputs/q_table.npy")

    # Karşılaştırma testleri için iki ajanı da 100'er bölüm çalıştırıyorum.
    print("\n[3/5] Degerlendirme: Q-Learning vs Random...")
    q_eval = evaluate(env, agent, episodes=100)
    random_eval = run_random_agent(env, episodes=100, seed=42)

    # Tüm eğitim sonuçlarını grafiklere dönüştürüyorum.
    print("\n[4/5] Grafikler uretiliyor...")
    plot_training_rewards(
        history["rewards"], "outputs/plots/training_rewards.png", window=50
    )
    plot_episode_lengths(
        history["lengths"], "outputs/plots/episode_lengths.png", window=50
    )
    plot_success_rate(
        history["success"], "outputs/plots/success_rate.png", window=50
    )
    plot_comparison(
        q_eval["rewards"],
        random_eval["rewards"],
        "outputs/plots/q_vs_random.png",
    )
    print("Grafikler kaydedildi: outputs/plots/")

    # Final bölümün GIF kaydını üretiyorum.
    print("\n[5/5] Final episode GIF'i uretiliyor...")
    record_episode_gif(
        env, agent, "outputs/gifs/final_episode.gif", fps=5, max_steps=300
    )

    # En sonunda bütün sayısal sonuçları konsola yazdırıyorum.
    print_summary(history, q_eval, random_eval)


if __name__ == "__main__":
    main()
