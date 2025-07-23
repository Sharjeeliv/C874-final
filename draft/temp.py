from collections import defaultdict

from surprise import Dataset, SVD, KNNBaseline, BaselineOnly, NMF, NormalPredictor
from surprise.model_selection import KFold

models = {
    "SVD": SVD,
    "KNNB": KNNBaseline,
    "BL": BaselineOnly,
    "NMF":NMF,
    "NormalPred": NormalPredictor
}


import math

def hr_ndcg_mrr_at_k(predictions, k=10, threshold=3.5):
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    hit_ratios = dict()
    ndcgs = dict()
    mrrs = dict()

    for uid, user_ratings in user_est_true.items():
        # Sort predictions for each user
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        top_k = user_ratings[:k]

        # List of binary relevance
        rels = [1 if true_r >= threshold else 0 for (_, true_r) in top_k]

        # HR@K
        hit_ratios[uid] = 1.0 if any(rels) else 0.0

        # NDCG@K
        dcg = sum((rel / math.log2(idx + 2)) for idx, rel in enumerate(rels))
        idcg = sum((1 / math.log2(idx + 2)) for idx in range(min(sum(rels), k)))
        ndcgs[uid] = dcg / idcg if idcg != 0 else 0.0

        # MRR@K
        try:
            rank = rels.index(1)
            mrrs[uid] = 1 / (rank + 1)
        except ValueError:
            mrrs[uid] = 0.0

    return hit_ratios, ndcgs, mrrs


def precision_recall_at_k(predictions, k=10, threshold=3.5):
    """Return precision and recall at k metrics for each user"""

    # First map the predictions to each user.
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = dict()
    recalls = dict()
    for uid, user_ratings in user_est_true.items():

        # Sort user ratings by estimated value
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        # Number of relevant items
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)

        # Number of recommended items in top k
        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])

        # Number of relevant and recommended items in top k
        n_rel_and_rec_k = sum(
            ((true_r >= threshold) and (est >= threshold))
            for (est, true_r) in user_ratings[:k]
        )

        # Precision@K: Proportion of recommended items that are relevant
        # When n_rec_k is 0, Precision is undefined. We here set it to 0.

        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0

        # Recall@K: Proportion of relevant items that are recommended
        # When n_rel is 0, Recall is undefined. We here set it to 0.

        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 0

    return precisions, recalls


data = Dataset.load_builtin("ml-100k")
kf = KFold(n_splits=5)

for name, model in models.items():
    algo = model()

    p, r, hr, ndcg, mrr = 0, 0, 0, 0, 0
    for trainset, testset in kf.split(data):
        algo.fit(trainset)
        predictions = algo.test(testset)
        precisions, recalls = precision_recall_at_k(predictions, k=10, threshold=4)
        hit_ratios, ndcgs, mrrs = hr_ndcg_mrr_at_k(predictions, k=10, threshold=4)

        p += sum(prec for prec in precisions.values()) / len(precisions)
        r += sum(rec for rec in recalls.values()) / len(recalls)
        hr += sum(hit for hit in hit_ratios.values()) / len(hit_ratios)
        ndcg += sum(n for n in ndcgs.values()) / len(ndcgs)
        mrr += sum(m for m in mrrs.values()) / len(mrrs)

    print(f"Result: {name}")
    print(f"precision@5:\t{p/5:.4f}")
    print(f"recall@5:\t{r/5:.4f}")
    print(f"hit@5:\t\t{hr/5:.4f}")
    print(f"ndcg@5:\t\t{ndcg/5:.4f}")
    print(f"mrr@5:\t\t{mrr/5:.4f}")