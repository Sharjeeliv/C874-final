from recbole.model.knowledge_aware_recommender import KGCN, KGIN, KGAT
from recbole.model.general_recommender import BPR, LightGCN, ItemKNN, NeuMF
from recbole.model.context_aware_recommender import FM, DeepFM, WideDeep

MODELS = {
    # 'BPR': BPR,
    'LightGCN': LightGCN,
    # 'ItemKNN': ItemKNN,
    # 'NeuMF': NeuMF,
    # 'FM': FM,
    # 'DeepFM': DeepFM,
    # 'WideDeep': WideDeep,
    # 'KGCN': KGCN,
    }