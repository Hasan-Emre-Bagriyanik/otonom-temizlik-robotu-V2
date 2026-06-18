import numpy as np

from .agent import RandomAgent


def train(env, agent, episodes: int = 3000, log_every: int = 100) -> dict:
    # Her bölümün istatistiklerini biriktireceğim sözlük.
    history = {
        "rewards": [],
        "lengths": [],
        "success": [],
        "epsilons": [],
    }

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0.0
        steps = 0
        info = {}

        done = False
        while not done:
            action = agent.choose_action(state, training=True)
            next_state, reward, done, info = env.step(action)
            # Asıl öğrenme bu satırda gerçekleşiyor.
            agent.update(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward
            steps += 1

        # Bölüm sonunda başarılı olup olmadığını kontrol ediyorum.
        success = info.get("result") == "success"
        agent.decay_epsilon()

        history["rewards"].append(total_reward)
        history["lengths"].append(steps)
        history["success"].append(int(success))
        history["epsilons"].append(agent.epsilon)

        # Her bölümde log basarsam 3000 satır çıkıyor, bu yüzden 100 bölümde bir basıyorum.
        if (episode + 1) % log_every == 0:
            avg_reward = float(np.mean(history["rewards"][-log_every:]))
            avg_length = float(np.mean(history["lengths"][-log_every:]))
            success_rate = float(np.mean(history["success"][-log_every:])) * 100
            print(
                f"Bolum {episode+1}/{episodes} | "
                f"Ort. Odul: {avg_reward:7.2f} | "
                f"Ort. Adim: {avg_length:6.1f} | "
                f"Basari: %{success_rate:5.1f} | "
                f"eps: {agent.epsilon:.3f}"
            )

    return history


def evaluate(env, agent, episodes: int = 100, eval_epsilon: float = 0.02) -> dict:
    # Değerlendirmede çok küçük bir rastgelelik bırakıyorum (eps=0.02).
    # Tam deterministik (eps=0) yapsam ortam da deterministik olduğu için aynı bölüm
    # 100 kez tekrarlanıyor; tek bir loop hatası bütün eval'i bozuyor.
    results = {
        "rewards": [],
        "lengths": [],
        "success": [],
    }

    # Geçici olarak ajanın epsilon'unu değerlendirme epsilonuna ayarlıyorum.
    saved_epsilon = getattr(agent, "epsilon", None)
    if saved_epsilon is not None:
        agent.epsilon = eval_epsilon

    try:
        for _ in range(episodes):
            state = env.reset()
            total_reward = 0.0
            steps = 0
            info = {}

            done = False
            while not done:
                # training=True veriyorum ki epsilon devreye girsin, ama eps çok küçük.
                action = agent.choose_action(state, training=True)
                next_state, reward, done, info = env.step(action)
                # Burada agent.update yok çünkü test aşamasındayız, Q-Table değişmesin.
                state = next_state
                total_reward += reward
                steps += 1

            success = info.get("result") == "success"
            results["rewards"].append(total_reward)
            results["lengths"].append(steps)
            results["success"].append(int(success))
    finally:
        # Orijinal epsilon'u geri yüklüyorum.
        if saved_epsilon is not None:
            agent.epsilon = saved_epsilon

    return results


def run_random_agent(env, episodes: int = 100, seed: int = 42) -> dict:
    # Random ajanı aynı testten geçiriyorum, baseline olarak Q-Learning ile karşılaştıracağım.
    random_agent = RandomAgent(seed=seed)
    return evaluate(env, random_agent, episodes=episodes)


def print_summary(history: dict, q_eval: dict, random_eval: dict):
    print()
    print("=" * 70)
    print("EGITIM OZETI")
    print("=" * 70)
    print(f"Toplam bolum: {len(history['rewards'])}")
    # Sadece son 100 bölüme bakıyorum çünkü ortalamayı tüm eğitimden alsam başta kötü değerler karışıyor.
    print(f"Son 100 bolum ort. odulu : {np.mean(history['rewards'][-100:]):.2f}")
    print(f"Son 100 bolum ort. adim  : {np.mean(history['lengths'][-100:]):.1f}")
    print(f"Son 100 bolum basari     : %{np.mean(history['success'][-100:])*100:.1f}")
    print()
    print("DEGERLENDIRME (epsilon = 0)")
    print("-" * 70)
    print(f"Q-Learning ort. odul     : {np.mean(q_eval['rewards']):.2f}")
    print(f"Q-Learning ort. adim     : {np.mean(q_eval['lengths']):.1f}")
    print(f"Q-Learning basari orani  : %{np.mean(q_eval['success'])*100:.1f}")
    print()
    print(f"Random ort. odul         : {np.mean(random_eval['rewards']):.2f}")
    print(f"Random ort. adim         : {np.mean(random_eval['lengths']):.1f}")
    print(f"Random basari orani      : %{np.mean(random_eval['success'])*100:.1f}")
    print("=" * 70)
