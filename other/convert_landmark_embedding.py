import numpy as np

data_path = "data/landmark_embedding.npy"
paddle_data_path = "data/paddle_landmark_embedding.npy"
lmk_embeddings = np.load(
    data_path, allow_pickle=True, encoding="latin1"
)
lmk_embeddings = lmk_embeddings[()]
lmk_embeddings["dynamic_lmk_faces_idx"] = lmk_embeddings["dynamic_lmk_faces_idx"].numpy()
lmk_embeddings["dynamic_lmk_bary_coords"] = lmk_embeddings["dynamic_lmk_bary_coords"].numpy()
np.save(paddle_data_path, lmk_embeddings, allow_pickle=True)

lmk_embeddings = np.load(
    paddle_data_path, allow_pickle=True, encoding="latin1"
)
lmk_embeddings = lmk_embeddings[()]
print(lmk_embeddings.keys())
print("Convert finish.")