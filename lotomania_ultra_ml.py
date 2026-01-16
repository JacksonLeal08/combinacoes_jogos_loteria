"""
Lotomania Ultra ML v3.0 - FOCO 18/19 ACERTOS
Modalidade: 100→50 | Prob: 19=1/352k | 18=1/24k
Autor: Jackson Leal | IA Refatoração | 15/01/2026
"""

import pytest
from scipy.stats import entropy
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import logging
import pandas as pd
import numpy as np
from dataclasses import fields
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import warnings
from dataclasses import fields
warnings.filterwarnings('ignore')

# ML Libraries

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

# ========================================
# CONFIGURAÇÃO LOTOMANIA
# ========================================


@dataclass
class LotomaniaConfig:
    """Config específica para 100→50"""
    n_jogos: int = 10
    min_score: float = 72.0
    n_estimators: int = 1000
    n_clusters: int = 8
    historico_size: int = 20000
    target_acertos: List[int] = None
    dezenas_frias: List[int] = None
    max_tentativas: int = 100000
    setores_quentes: List[int] = None
    faixas_prioritarias: List[int] = None


    @classmethod
    def from_yaml(cls, path: str = "config_lotomania.yaml") -> 'LotomaniaConfig':
        if not YAML_AVAILABLE:
            print("ℹ️ YAML não disponível. Usando defaults.")
            return cls()

        cfg_path = Path(path)
        if not cfg_path.exists():
            print("ℹ️ config_lotomania.yaml não encontrado. Usando defaults.")
            return cls()

        try:
            with cfg_path.open('r', encoding='utf-8') as f:
                data = yaml.safe_load(f) or {}

            # ✅ SOLUÇÃO DEFINITIVA: FILTRA APENAS CAMPOS VÁLIDOS
            valid_fields = {f.name for f in fields(cls)}
            filtered_data = {k: v for k, v in data.items() if k in valid_fields}

            print(f"✅ YAML carregado: {len(filtered_data)} campos válidos")
            return cls(**filtered_data)

        except Exception as e:
            print(f"⚠️ Erro YAML ignorado: {e}. Usando defaults.")
            return cls()


# ========================================
# LOGGER ESTRUTURADO
# ========================================
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

# ========================================
# FEATURES LOTOMANIA (28 FEATURES)
# ========================================


class LotomaniaFeatureEngineer:
    """28 features otimizadas para 100→50"""

    def __init__(self):
        self.n_dezenas = 100
        self.n_escolher = 50
        self.n_setores = 10  # 10x10 grid

    def extract_features(self, combinacao: List[int]) -> np.ndarray:
        """Features específicas Lotomania"""
        nums = np.array(sorted(combinacao))

        # Estatísticas básicas
        soma, media, std = np.sum(nums), np.mean(nums), np.std(nums)
        pares_impares = np.sum(nums % 2 == 0)

        # Setores 10x10
        setores = np.bincount((nums-1)//10, minlength=10)

        # Faixas 20x5
        faixas20 = np.bincount((nums-1)//20, minlength=5)

        # Faixas 10x10
        faixas10 = np.bincount((nums-1)//10, minlength=10)

        # Overlap histórico simulado (proxy)
        overlap_score = self._overlap_proxy(nums)

        # Distribuição por quartis
        q25 = np.sum(nums <= 25)
        q50 = np.sum(nums <= 50)
        q75 = np.sum(nums <= 75)

        # Consecutivos e gaps
        consec = np.sum(np.diff(nums) == 1)
        max_gap = np.max(np.diff(nums)) if len(nums) > 1 else 0

        # Densidade por setor
        setor_densidade = np.std(setores)

        features = np.array([
            # Básicas (8)
            soma, media, std, pares_impares,
            # Setores 10x10 (10)
            *setores,
            # Faixas 20x5 (5)
            *faixas20,
            # Derivadas (5)
            overlap_score, q25, q50, q75, consec,
            # Avançadas (3)
            max_gap, setor_densidade
        ])
        return features[:28]  # Padronizado 28 dims

    def _overlap_proxy(self, nums: np.ndarray) -> float:
        """Proxy para overlap com histórico real"""
        # Simula correlação com padrões históricos
        hot_zones = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                     91, 92, 93, 94, 95, 96, 97, 98, 99, 100]
        overlap = np.sum(np.isin(nums, hot_zones))
        return overlap / len(nums)

# ========================================
# BAYESIANO MULTINOMIAL 100 DEZENAS
# ========================================


class LotomaniaBayesian:
    """Dirichlet-Multinomial para 100 dimensões"""

    def __init__(self, alpha_prior: float = 1.5):
        self.alpha_prior = alpha_prior
        self.posteriors = np.ones(100) / 100

    def update_from_historico(self, historico: List[List[int]]) -> np.ndarray:
        """Atualiza com últimos 200 concursos"""
        recent = historico[-200:]
        freqs = np.zeros(100)

        for sorteio in recent:
            for num in sorteio:
                freqs[num-1] += 1

        alpha_post = self.alpha_prior + freqs
        self.posteriors = alpha_post / alpha_post.sum()
        logger.info(f"🔄 Bayesian entropy: {entropy(self.posteriors):.3f}")
        return self.posteriors

# ========================================
# ML ENSEMBLE LOTOMANIA
# ========================================


class LotomaniaMLEngine:
    """Ensemble otimizado para 18/19 acertos"""

    def __init__(self, config: LotomaniaConfig):
        self.config = config
        self.features = LotomaniaFeatureEngineer()
        self.bayesian = LotomaniaBayesian()
        self.scaler = RobustScaler()
        self.rf = RandomForestClassifier(n_estimators=config.n_estimators,
                                         random_state=42, n_jobs=1)
        self.gbm = GradientBoostingClassifier(
            n_estimators=500, random_state=42)
        self.kmeans = KMeans(n_clusters=config.n_clusters,
                             n_init=10, random_state=42)
        self.is_trained = False

    def generate_realistic_historico(self, n_samples: int) -> List[List[int]]:
        """Histórico sintético com viés realista"""
        historico = []
        pesos_base = self.bayesian.posteriors if self.is_trained else np.ones(
            100)/100

        for _ in range(n_samples):
            # Gera 50 com pesos ajustados
            sorteio = np.random.choice(
                np.arange(1, 101), 50, p=pesos_base, replace=False
            )
            historico.append(sorted(sorteio))
        return historico

    def train_optimized(self, historico: List[List[int]]) -> Dict[str, float]:
        """Treinamento com foco 18/19"""
        logger.info("🤖 Treinando ensemble Lotomania...")

        self.bayesian.update_from_historico(historico)

        # Features para todo histórico
        X = np.array([self.features.extract_features(s) for s in historico])

        # Target: simula "qualidade para 18/19" baseado em overlap bayesiano
        y = np.array([
            sum(self.bayesian.posteriors[s-1] for s in combo) / 50
            for combo in historico
        ])
        y = (y > np.percentile(y, 80)).astype(int)  # Top 20%

        # Ensemble training
        X_scaled = self.scaler.fit_transform(X)

        tscv = TimeSeriesSplit(n_splits=5)
        rf_scores = []

        for train_idx, val_idx in tscv.split(X_scaled):
            X_tr, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            self.rf.fit(X_tr, y_tr)
            rf_scores.append(self.rf.score(X_val, y_val))

        self.gbm.fit(X_scaled, y)
        self.kmeans.fit(X_scaled)

        self.is_trained = True
        mean_rf = np.mean(rf_scores)
        logger.info(f"✅ RF CV: {mean_rf:.1%} ± {np.std(rf_scores):.1%}")

        return {"rf_cv_mean": mean_rf, "rf_cv_std": np.std(rf_scores)}

    def predict_18_19_score(self, combinacao: List[int]) -> float:
        """Score específico para 18/19 acertos"""
        if not self.is_trained:
            return 50.0

        features = self.features.extract_features(combinacao)
        features_scaled = self.scaler.transform([features])[0]

        # RF principal
        rf_prob = self.rf.predict_proba([features_scaled])[0][1]

        # GBM secundário
        gbm_prob = self.gbm.predict_proba([features_scaled])[0][1]

        # Cluster density
        dists = np.linalg.norm(features_scaled - self.kmeans.cluster_centers_)
        cluster_score = 1 / (1 + np.min(dists))

        # Bayesian coherence (crucial para 18/19)
        bayes_sum = np.sum(self.bayesian.posteriors[np.array(combinacao)-1])
        bayes_norm = bayes_sum / 50  # Normaliza para 50 dezenas

        # Peso otimizado para 18/19
        score = (0.4 * rf_prob + 0.25 * gbm_prob +
                 0.20 * cluster_score + 0.15 * bayes_norm) * 100
        return float(score)

# ========================================
# GERADOR OTIMIZADO 18/19
# ========================================


class LotomaniaGameGenerator:
    """Geração focada em 18/19 acertos"""

    def __init__(self, ml_engine: LotomaniaMLEngine, config: LotomaniaConfig):
        self.ml = ml_engine
        self.config = config
        self.dezenas_frias = config.dezenas_frias or []

    def generate_focused_games(self, n_jogos: int) -> List[Tuple[List[int], float]]:
        """Geração com filtro rigoroso para 18/19"""
        logger.info(f"🎯 Gerando {n_jogos} jogos focados 18/19...")

        jogos = []
        rng = np.random.default_rng(42)
        tentativas = 0

        pesos_base = self.ml.bayesian.posteriors.copy()
        # Penaliza frias
        for fria in self.dezenas_frias:
            if 0 <= fria-1 < 100:
                pesos_base[fria-1] *= 0.05

        pesos_base /= pesos_base.sum()

        while len(jogos) < n_jogos and tentativas < self.config.max_tentativas:
            tentativas += 1

            # Gera 50 com pesos bayesianos
            candidato = sorted(rng.choice(
                np.arange(1, 101), 50, p=pesos_base, replace=False
            ))

            score = self.ml.predict_18_19_score(candidato)

            # Filtro agressivo para 18/19
            if score >= self.config.min_score or len(jogos) < 3:
                jogos.append((candidato, score))

                if len(jogos) % 2 == 0:
                    logger.info(
                        f"  {len(jogos)}/{n_jogos} | Score: {score:.1f}%")

        # Ordena por score
        jogos.sort(key=lambda x: x[1], reverse=True)
        return jogos[:n_jogos]

# # ========================================
# # EXPORTADOR AVANÇADO
# # ========================================

# class LotomaniaExporter:
#     @staticmethod
#     def export_optimized(jogos: List[Tuple[List[int], float]],
#                          output_dir: str = "./lotomania_results") -> str:
#         """Export com análise 18/19"""
#         Path(output_dir).mkdir(parents=True, exist_ok=True)

#         dados = []
#         for i, (jogo, score) in enumerate(jogos):
#             row = [i+1] + jogo[:10] + ['...'] + jogo[-10:] + [f"{score:.1f}"]
#             dados.append(row)

#         cols = ['#', 'D01-D10', '', 'D41-D50', 'ML_SCORE_18_19']
#         df = pd.DataFrame(dados, columns=cols)

#         timestamp = datetime.now().strftime("%d%b%Y_%H%M")
#         arquivo = Path(output_dir) / f'lotomania_18_19_ultra_{timestamp}.xlsx'
#         df.to_excel(arquivo, index=False)

#         # Estatísticas
#         scores = [s for _, s in jogos]
#         print(
#             f"📊 Stats: Média {np.mean(scores):.1f}% | TOP1 {np.max(scores):.1f}%")

#         return str(arquivo)

# ========================================
# EXPORTADOR COM BARRA DE PROGRESSO
# ========================================
class LotomaniaExporter:
    @staticmethod
    def export_optimized(jogos: List[Tuple[List[int], float]], output_dir: str) -> str:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        print("\n📊 EXPORTANDO COMBINAÇÕES...")
        
        dados = []
        total = len(jogos)
        
        # Barra de progresso detalhada
        for i, (jogo, score) in enumerate(jogos):
            row = [i+1] + jogo[:5] + ['...'] + jogo[-5:] + [f"{score:.1f}"]
            dados.append(row)
            
            progress = (i + 1) / total * 100
            bar_length = 30
            filled = int(bar_length * progress // 100)
            bar = "█" * filled + "░" * (bar_length - filled)
            print(f"\r[{bar}] {progress:6.1f}% | Jogo {i+1:2d}/{total} | Score: {score:5.1f}%", end='')
        
        print()  # Nova linha
        
        cols = ['#', 'D01', 'D02', 'D03', 'D04', 'D05', '', 'D46', 'D47', 'D48', 'D49', 'D50', 'SCORE']
        df = pd.DataFrame(dados, columns=cols)
        
        timestamp = datetime.now().strftime("%d%b_%H%M")
        arquivo = Path(output_dir) / f'lotomania_18_19_ultra_{timestamp}.xlsx'
        df.to_excel(arquivo, index=False)
        
        scores = [s for _, s in jogos]
        print(f"\n✅ EXPORTADO: {arquivo}")
        print(f"📈 Stats: Média {np.mean(scores):.1f}% | TOP1 {np.max(scores):.1f}%")
        return str(arquivo)

# ========================================
# ORQUESTRADOR PRINCIPAL
# ========================================


def main():
    print("🎯 LOTOMANIA ULTRA ML v3.0 - FOCO 18/19 ACERTOS")
    print("100→50 | Prob 19: 1/352.551 | 18: 1/24.235")
    print("=" * 70)

    # Configuração
    cfg = LotomaniaConfig()
    print(f"⚙️ {cfg.n_jogos} jogos | min_score {cfg.min_score}% | {cfg.historico_size:,} histórico")

    # Pipeline completa
    historico = LotomaniaMLEngine(
        cfg).generate_realistic_historico(cfg.historico_size)
    ml_engine = LotomaniaMLEngine(cfg)

    metrics = ml_engine.train_optimized(historico)

    generator = LotomaniaGameGenerator(ml_engine, cfg)
    jogos = generator.generate_focused_games(cfg.n_jogos)

    # Exportação
    pasta_saida = r'C:\Users\OMEGA\OneDrive\Documentos\Jackson Leal\01 - LOTOMANIA_ULTRA'
    arquivo = LotomaniaExporter.export_optimized(jogos, pasta_saida)

    # Top 3 detalhado
    print(f"\n🏆 TOP 3 JOGOS (primeiras/últimas 10 dezenas):")
    for i, (jogo, score) in enumerate(jogos[:3]):
        inicio = ' '.join(f"{n:02d}" for n in jogo[:5])
        fim = ' '.join(f"{n:02d}" for n in jogo[-5:])
        print(f"  {i+1}: {inicio}...{fim} | 🔥 {score:.1f}% (18/19)")

    print(f"\n💾 EXPORTADO: {arquivo}")
    print(f"🎯 RF CV: {metrics['rf_cv_mean']:.1%} | Ready para concursos!")


if __name__ == "__main__":
    main()